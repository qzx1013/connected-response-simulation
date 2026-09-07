"""Lean streamed single-site BP engine for cross-layer second-order terms.

This is a self-contained sibling of ``plaquette_bp_streamed_optimized.py``.
It deliberately does not import that module, the legacy single-site engines,
or the package in ``streamed_single_site_engine``.  The physical PEPS
evolution, dressed-Kraus insertion, exact retained-mode/all-pair accounting,
and public scheduling API are kept, while the environment is consistently
changed from disjoint 2x2 plaquettes to single-site Bethe BP with a local
cluster used only at readout:

* one rank-2 directed message per lattice edge and branch;
* in-place Gauss-Seidel single-site message sweeps;
* site/edge Bethe normalization outside the readout cluster;
* exact contraction inside either the target support or a 2x2 local cluster;
* selectable ordinary single-site boundary messages or transient two-leg joint
  halo messages adjacent to the target.  Dense three-leg messages are
  deliberately not formed because they scale as ``chi**6`` in the doubled
  virtual space.
* selectable Bethe-ratio, source-anchored, source-anchored ratio-of-sums, or
  unnormalized full-Bethe readout.
  ``readout_block`` selects the support-only belief, a minimal-support
  correlated strip, or a 2x2 pair cavity.

Only scheduling and the explicitly selected BP approximation differ.  There
is no sampling, CPU offload, or hidden approximation in the pair ledger.
"""

from __future__ import annotations

import base64
import gc
import hashlib
import json
from dataclasses import dataclass, replace
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Iterator
import zlib

import torch


DIR_TO_AXIS = {"U": 0, "D": 1, "L": 2, "R": 3}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}

_EINSUM_PATH_CACHE: dict[tuple, list[int]] = {}
_EINSUM_REFERENCE_PATH_CACHE: dict[tuple, list[int] | None] = {}
_LABELED_PAIRWISE_TREE_CACHE: dict[tuple, tuple[tuple[Any, ...], ...]] = {}
_GLOOP_REGION_CACHE: dict[tuple, tuple[tuple[frozenset[int], int], ...]] = {}
_GLOOP_CONTRACT_EXPR_CACHE: dict[tuple, Any] = {}
_LABELED_2X2_BOUNDED_TREE_CACHE: dict[
    tuple, tuple[tuple[tuple[Any, ...], ...], int, bool]
] = {}
_LABELED_2X2_PERSISTENT_CACHE_VERSION = 1
_LABELED_2X2_BUILTIN_CACHE_ENTRIES: dict[str, dict[str, Any]] = json.loads(
    zlib.decompress(
        base64.b64decode(
            b"eNrsveuO67ySJfgu3+/8weCd9SqnDwq8NgrTqG5UVw1m0Jh3H94kkRQlS047d+795bnktuyQRMlyMBixYq3/8xdCmILG2CJpIUCg"
            b"VjumA7aCSxMk4YAs8So4HKhj3CsVrTySkpmggkZ//cv/+Uvb//y3//nv//uvf/nHP/76D+/+y/q/PvDHP9gH/xAf8kP9858f2wf0"
            b"4x8o/gH0AfAB+ANI97H8+If6gPg5+wD+AaL7kMSjig8MHxh/YPKBafcpz4eOR8XoA7MPzD9wvzvgePR4YEzjJx9xA5eh/S/9b/8R"
            b"P+VE0niKeOEfVAHD+Wg4DSMdqjWlSPEPCSp+iLGUyZDEkYOKo2qPGIcR92QiGeBoQj/SPSGotUFpNIjK+LYop5T58uIwZXvKDxbH"
            b"wuJgyu2r9y7dqu548SaUwVOFi3E15cttEf214DRwtH1b472vVvEi0smFIMvFpB2aYTTGySgeUsjVNh94td0fnuD85QBH6QtOX9H6"
            b"Hc++qHrbARNEl1PIfONQ3rP9Esp9yDeXLV/WekvqLnh+GlT3YJjSvJ9Y9mTNzsP5yqDKLpyK8hgN9yDt3t6veKOKKWeMsDxCkm/Q"
            b"Pz/++u//4b37f/81eP2//838D//Xv/znf/yX//jrf/0P/e//7t2//i+v/6+//gXiA8REHPBf//vf/vu/6//8r/+Ihn/947/9Zf7n"
            b"f/27i2b4/8H/+p/xSP/6f8N/++vjH/HbpnE4ND8WcSDdBksPkcxXmLf7LRY3ymax7rfYukXrp+1WOms8zXIOvrwux8gvi3ndyNaT"
            b"9yH9tzOJr9NdW35ladTF9+TXqLobWh1O9AL5KVf1QSxfEZRfcPpK/5n9THUyZbt4FpY/z56luBKZDVR5I40g/76XU9J1I/+oSTlQ"
            b"/m0vG+nJJbgYJf+xbuQHlJCyR3JAaeOf28PxkU4hiKAg4xD+UX4zaRT/gHxbl794/YvzO8tfvP5FH3Vc8S/Jr0ndC2UblF+T/DoO"
            b"4Z9//X8fcd6gnDlko/tSBjlqqbACOBfBmzh/KOwtUGkwA6GcJSTOG5gL4QKzTDJEPzlvzCYNUnxX7/J5/93OppPiAMh8Lkl78fks"
            b"QgZ/kR++5ilsr6K6uTohqG1CKA9p58Lzzw2Xy1XLBa9XtniZMh9gutpBeqK7A6UTJYuJI2a9P413ong4nHxcuVd48Gu7iWrxubgd"
            b"e/2YQLyt7TmHqTENKdsQhchyq9nOLH8avwuieJ5E623nnUn8KH0ncVJm62/luuukl30mfCz/LT+O/euXfvLg1Wc8Hlk8nth+GH1k"
            b"hVaPR/PnKZyqHk9kgzxZnnm87Negd36bL9t7L6hn/ZT32jzW4qU4KMe9xiQQTxkCr2M4y5RigVtgXMjoxJwX3lNpmabCIKlZkC4Y"
            b"FLQ78lLQRHSdc8geReaf7dR7tcHSLuTdxbvVC+F10lliwt4X8TwHyTxF9L9/nr1Eimi3wHKIZQ9+18MPEScXQWia0LNL2ubG1pPI"
            b"EsURvlyROBjZJIDOk1q6xD48ru4ELXFnCqH3w4sjS2cFgXLIgZaTl1Gy6RDWIBIrvH07dDaKfPocRUpVTNlhvFmOl0IbTNgSrZej"
            b"li9RjseOx0zfEJFloVAOXo9cLmD8ygSqRwfMcD1HvmyMmmse9srHLzukpU6+nyzvBPlbvOowCb4TZiYXhFfHtr1On8jyaf5ke82q"
            b"Fa2RKW4+KVawhJbr63i1yW4XHc7fTTEjXd+kJ64UVj8I6dErv93sBuPToLL/pA+jR7bEc7iswfKvuPxsVTIgqLwxRokl5nsQJeZf"
            b"zRIlpsd2jRJzQJi+XjY6W6DlhKvr3Bzo4kbPXXCxRKvDRTk4hCY49MwiiqhjhmusjELKEKSwQUxjr0F4hZFH0scPCA8WI6IZJ5ww"
            b"66j/g5IKowtOa2bUr5avZgBu5CduLacvJzPSiQVXfTJjvJot/QCr96sh+2ykuPHCqJy6fMNqHzLGUxczIGgJoWkxLQ5/HDHPC9ka"
            b"h6YhlKuC/ousUxdaAtdlxV6Df1gmTdF/HeX2Qvpe8xDq/H2eucgDT/sAwVgsl5CGn/fdzjfbN+2RhxefBKG6e1vOzHcJrWhWvohn"
            b"EwsYKRFPecfjp0xCykJVJ91sJH+cNzit7qTdWlISnDYpifWznITgtE1JcNqlJOjq8en6Ht+9rGmHXX6hvH9tVvhJKbwnpQBgGEHa"
            b"Iau0EXFGcBxZ4WLIHmyQwQTJMVjNcSAgGfVYQaBMxNnGayEOUwp0lzK9Pjms+YPlW5zMHG0cPI3cH4TtD/LB03B8Mq0Mq/rLIfYa"
            b"2vOPJbjfR/ZLCjVGq3LLuE7CX77F1WSbosgS/45Xm43yXIVp8dTVnC970GE1IOUahkMfK5Mhuk8HTH+b758eHTib5VEo3GdUJqaq"
            b"3PjokTlaxpHCwdE0j7NYRV9MVYnKym3OAx6OnG3SUuYz4Xm8wDsJDdykHfav1x/5pU/q62YL7xIaOSK/kOQodt8nMk9OePXZKdH2"
            b"zrj8NSmQmYel2jLCpXDGa88dJUzQQARylCFJEQrCauGFwJgDx9h7Jhx2XCokJGen6RAoq/MxHVLDufvhOlmd8twji+OMLpwF8ks6"
            b"6zCQn4Zuq5dk5ae+//leCfV58e8Q10b9vFKzY6ufyxbxD2x+7sA02SSfgQRrfVG1zs/5sEM0zO5LUtImmenxPrxOSRhKXnrJi9cb"
            b"1R8/HZamf3J8q7pS5fGoknkaE2NNga7MyEf75NGkbw3jcjubHPK2em7sk1mZEREleNyDQPrxdnc2pcXTNAqUYTkbFxmvIg6/GK8+"
            b"vE4KV913Ht9VBy7L8mF1xttreZhEkSVXVO3a1+vReHs4vn5WUylyC5tlqdxtb66vcrS9FP84fXO2mjbB9PJgbsE0uRJMi/rL2gXT"
            b"NWjGbdCMayY7f7XTTHYNmrcUSevSceOiW5eOajKlhMtoDZ1hdfWLG2cKOxe8UUoZpp1SnntFgTgjbfLpjmmsHASFKReGBw1eB4wY"
            b"psJR/7dKr1zDN1xMgtCahUl/5T67Qcg1Z79WQ9p8TQ2acU2DLFWzcuP6hBFF1RII4V2dLT+So8MvmRtRbxErh0az2wD1iOlvM/Uc"
            b"HTlbpQNDfPaajARrw+xhn2xZ7h+JD6RY3GrBhWz79DdzMgM1NZHhFOmNnCYjWCq+1A1pY07GaaRYEoaQ3OaF+VDKqIvpJ9MqNxz9"
            b"MVyjunre+voNflENt1mh2Shlii4bs251bnx4WX1/k0qZO/0j+5/MypdnVgJoiqUxkgAzCmsm4yvrEogDiGWK40CxjitbLbQD7hhP"
            b"Mwjz1mNvwreK+4+C/vOI/1G438f6BabROEw1liXTp7jPQvcWY/J5fRR7fEbKOy+uU5RoeXVsfZ1wl9Onu4GT7FrjoZSaJDGWB/MB"
            b"/HBaUZBr+gfVqB5v39QOHlns4h86QbHsqxXJLs3ihCk+jHs/4nRt2XDFe9wLsOEmSg43sfL2upYWPxbX2RYpixVtEiTtJ7RJnWwJ"
            b"7A0XtyRI7pQpv0kMXZzgUL18e6S8i44xpUFZH6dYpQLX8YvFAUOIf5BnSoZgLfGaWWbiR8xioq2xNuHVPHdCP+PsasVqjvk4zTsv"
            b"oRLIQ28HC8RqcHi10H+Qd4YHiBF2BTHyorxyCxlhfbiZL77LcUCPKJt4KDUDRczwHHAM6ShmBUOMWqeDmxWC3O2WsGilTEj48syv"
            b"2aBJ0U6U2FuyBdnBm5zDcKUpi5M9PF7vWfqi8Iq3GmeYOgyCVf4aln3wkp1YSondbmkseQ+KJFlGpZo5OV/6uFc2Lveqz06nmyXv"
            b"eF8hBL6Xo87wxBUp0m99LJFnF7TWz0i5E0ugPGymY5AucG63oEUcw4ZQHt4nuOyVN5qAGB4Dky9ksOU/dwvHf7Ylp7K9ZrBhzGCz"
            b"CxnsLo5d8xsPc9hQwjg5OvX0xMm3Jj8ws8EbHP04kxAQAUSEB+SY18FzhIFoobhnVsUBUYRNdPKMK8xNfHwJfwPwuPiLz6Q9jnMe"
            b"8iFw7mp98GnYyZ1ui4NsS5dqqekIvKWr6/q/L/Uli0m7yWC3c8gwNHVcg8XsqrAF3SHqpchlFjncId+cPAhglKLmPsEACRkbZbJ5"
            b"dOZAtvpgCa0P9yvWGQa4OONuDr7sjeOPR+Abvrj3oKM/XZb+0LV+wOZPaU5jVDfebKTPsoddPms2VkQHrCgP2LIKvWdeq4clnP7l"
            b"eYYlMG5cbwmYO1w0bvMMeMszkNY/b9mE+B1D/ponOYX3JBQWx8sBU04xYy5EL6yDcNZC4GAlNoSAodR4I4ixxHrBBbUkBK+0dRw5"
            b"bN+TdT7IFnyy2nfkZoduh+fcrFggaE+i+y4i8c6gdWPXxj3Q3mMk3nj8fNx8YfxSSmAByqUXgjXZkQXZXX+Tw0kqiEthLIfKJlse"
            b"hCFjUmxz+4ha4dodoHDfiZkNa0Mh3Z1pRfGR/nriZRRzzlhpeREtyLud0rIBaboB76Y44lmu+/SjciAcIrTT67agiIdP6FBQxKPP"
            b"XrHYLR5kQIbsLNcMc3n3ty4qtnnkdhI4ySM/TqF8QR4ZC6yJ1hYhoMJprmOYbZAUCpRGoKWhXlkhgDMkJWZCKEqUZAI5T1yw1xB6"
            b"J/C829i8EyRIxeYdAvPEOUZuDwAhS40KSVzW49W2OoQRvvA41yJrqMwXA7Wi2PqAOloI2WHR8pPULftRTkhT2lnx5ibUc9KShlBt"
            b"gL7HtuWLVDklzNZDyuVq9ziVbFZQhGhLpOUdBuhLtoh/YJmz9kb50zyzLklglpNvNxzk3R6WDcu2vS4pWrz6zra7hX6U3ET5ZHtd"
            b"PqHrPm0SeN/WvL67tUm/GxO3oh6GsJafls+W5tOx86/ByNQvaYJ5e0UpbB+tKuEMtiBtDEsBacW1o4RoFUBJJQg3FiMmDSPpGXLC"
            b"xJhVx5faGKGY/sWdf5fSB/gBaoIfoSbqIn3BAdA9DuASCvk4FXzej8LPmyRyPLuu+if8Ejl9na4ZT3AVQ7Y2d0kLvu+j7C3Z2mdY"
            b"uzJYacLeYGtkSA+ssXVlhii3ktS7ua9xCVGtgRHBlpOIPmO7T2uni8wlPYLX/uwWEyemvY2kXDRUOEPe6+QkZUjZusvu3gE03PKn"
            b"c5ww+1gQycsnbbfgHKl89slWSttebd70+NX38bCyPtJNAHnsY5uc7Qyt9sj7vq0ZEFunLGeORe8bQAJxXCNvDI8+2mpwInhJMQpe"
            b"ecKRIQgCNnGH+Cg6b8gDTzz1wy/ywg17xFmEid/mn4+8c+2zlo2NOKI12icgpj46Ha34QAyokt0sDq3BTvVePbthuXDjyCXMHYAQ"
            b"OGO96OKJKqzqOUedyXWimVygFdmzHQ0wXkEyjBPVAsRAswklf56DSbr15PVTUvrouWCz5HSuI72gW0+362yo8ea2nsbbCpy3mAM+"
            b"wH3pR9tU11PzwADk3QN9fwN/WP3eDHvw2O+hofy04ArOC1ht7LkVpxxHmiopqBUKueC9iPGlC2BQnF0FMK2xxDghc4kgQRFptOWS"
            b"OCDICfJHtLBdWNheiy/3YSPZvvVnyCR6TNZauhcj1dq+QIV7eqE1w5pbu3CTFzgcpWxCRl4xVFvy+hHxBRxTXySLhLaKQVzbFFcx"
            b"ILOiE66DiM/c8hU1SYM98iwdudimniBo29hqLD8piGXLdG8+RzZBGKMpnL3uP0tj8IaA7bdoU3dqX9d24gwCqB3GzVYGzlZ8wLJu"
            b"37bmbRA7GGx9uTYjX2qP+Olze6bPDQcclInRpCfUaiSQDhJZE+NNL7nD3irlA2NS6rj0Jyp+gMBogYLhVHDzJ5KT1UU1LMvqdqgH"
            b"tfuJx7tS8DpqNyD9sVi2ecxNtuNwyDHCbWayPcHDUp0vt5jv2MmiHVGlACMWEC0eyccIy6nh9/GTpSIQv1XOmTu4kp+VTaFHNqWZ"
            b"Y+dWCj9tVb8t6jdFnfnLFSD1w1d2JWtJiOSc0rg8djY9OcyFRF/GUXrSHLIpcFQxxoxTIiIaW2sEUCm8CYkHIXxpjf2K+8KvxOVf"
            b"pCCchIMXWXLwDFR/6IJGMP9D1/jQBfVHnCFaT6ryl2EBcy/fl4sqz0yazWHHYNv1E4y4gOgsxcBvuTj/oSIlVL00WKo9w0n2oAAB"
            b"ZUDPdtfeDiezA2uarfotOOXCXT0nbT1nX04flvZ7TGkDb2oL6C0Gtdt4AR/uV5TGZ9720922bwomCdVMBUwkaI81M4ZbQAhHj8wd"
            b"SKcCCij6HKuZFcJbF9f9xnpGFddKuacqSezEVR/76a9vv119Sf7VsifrSJfrOViUQvyWS+zIAcY8YXG2RPBJhyw6qTi1d3gtD6XD"
            b"xNm5rn47QvPJIbMl5N4uaIgDyK43Fj0zcw3pUjipa6UqUHxTIDr2tU56q5LVMucohUfGg+l1KqjGPRT1PWgluVSLmgoR3rzrtHYk"
            b"DytEZ7nWpWK0e8XXkzUvt0X+N+INu188Ona+j7pY31Y8ShO2ExQJzbEk1utAWXyLayet8JRb6RWhgOPPkwVjQtDSB6E9p9h6x3+a"
            b"uW7mYQ+aq7646asnEmY1FJlf5hnjGMwYd+lKO7YR9EID8N/vcbeNq0b5iLdtWWrprhij5tSQlWyJLHNF15DFGr7Kns03O95MAsxx"
            b"4dphHd5VHNyvYp47Fb64k+vT/VgnfWA1MK+9X/3WEFLvshN8ay/gdMSaFkzxZ3O0P51ck04u4kLqIXDBB+wlMgYbYyTnXAfJBPVM"
            b"q/iwBk694oxjIbnmzFjrPAdL4LVo0nuIq/Ny/hPsure6rGq3UXZhpItut96nnrMwmRU2mguMMRcmjiXy5x8NiGxY4O9zzu03siaB"
            b"c7K4b/PCu8TzXlJoyysPSNdyhXlSA0aR3F/srvcim6V5hFHRZzvw7DstZnmGYhyhYYedtE8xehZYxfi9/qwp9cDSJwtNFWyLlOeo"
            b"gpIwxk3CGHc9WacJiqErq3er3x7Beo8A5u3Y1tOuLEo08zIGyNRhi4LwOp48nh6pJNXmHfbeU3BaSU+cRMRqTgIySMpgEdM/lI4n"
            b"lI6XF+R9vLqCQMmzlPAXeSJv8TJe55S8m3j+CgrKdG/zZWJclIrU2g084FpHMrB0DXknoIyiWYfXnr+ymD6dZcZ3hdhm/Vesdk/R"
            b"ho1x08HoqR6h6Ts4OtoRxUzbW9C/9xVCQ3836kZKXFo4WeFMMIoqHjxQS7wnljscCJdOOTAaeYQTzaO21jsvGXbSEWTeARB7b6/B"
            b"U0mIuQTmtVCa0439j+AhbN866tYioKx1QIIVR42L30LFgdfxoM42UVXrGdPbtPZ4bfcYIPMxKwPktRC3XFxpMqaKUDqlWt/thuvx"
            b"419Y49wltTPdYzl8+isPiNdnLbU0N/d3PbhtrnofsQOq5gmxw7fvgq+Zk6EyWezWntp7YTint6Lwud+GGlBt+ebtNW16ZWnXN3sc"
            b"0zdR+LSjdsarvnh3WOnF4GHe+reMyvd5j66d9uvj9UllkVIlFNI6huoIUFJF8jxgKbzF0kqQHBwNnMTpwDgD1lCOYmDiqJceKU3+"
            b"HB7fq7Q2F7WKr4DUErc3btld2E7Ahz7Jz3Cd+PcWre0WlS/yF0eY4AOKnX3jwwGFelcg3FGol8tnQA4rhDCyFRdjhtcutpFIfcJy"
            b"nC8zg5UJ4rA0zPFOSXrfmkeX0ySuazrZa6VoGPgZ4inKLs9y+t5sKZ4XHs84Io+43s+OtrW2zYL+vi2OTl61e/zQ9n41bS9lAnCg"
            b"NEZ6LDp8jYSmGDubiqLGMKapA6xjMGgCBa8lIdQgRbWMywcL8gcNeBnkdznTc9yXIXclzE4fbk0KfQZkmE+bz862s885fbJNmgoJ"
            b"IdBR9MgdriabZAbempA5p/ORW/sz4bJz6HwZ+MhMXFRBSFUFwQvX8ebHx+Eny3T3KAI2OcXsK0wPZR7Rs4mb/DTd6TSZc93Iw/yM"
            b"PIzmZV0nyIYXQrbdJXx8ucTvLQKlgaVUyx8g4OuAgJQHGx98SzSiQkRPizkREgVPKTNcGhsEcjg6YMQdA2uJ8lILZw2LL0T4IubJ"
            b"T9LiwCEtDtvR4nRQDtJllLtoNBXQG7kMsiOdiceIBpItR2ILZrg26B1l7Q+U1A50qMWBYvTI8EgGRhyWzTBbh8c7todJOHlRxG9h"
            b"nSBVBqOuVOph7y1p8kHy6qT2mdwMX29lOI4YceBQ73mx2hhxcJP7mDPiHDGBjcmO768ocUiJ884wc18qtE5gqlWcuq3kjlNhVNwA"
            b"poRP8ppccIqMtCio6M+w1VQjzXIzM4mf/8mUOBd/sYvvaUjA8KWq4Zxy4Zq4/ZE3213EHZn6WY1v5UrfCzRkXUoSo6seoHaANkmH"
            b"5LQD2NEl+7vnYByd25yGuEDdkstuNToP0sN7jU5aI9/9cPNVrSqdz5Hf0HsQ5rnzrASLzScbuPlIh+JczQ1PwsY+WKQHwOYfCpyX"
            b"opipw4F5IzR4KmPUCJgZZqnT3DpKndSBEw9BWwNMBBTARI9MBJZBG8vCD4r5ARjtMej41UDnHxTzH4FilkRKjm6lbs9RzOwUxcxO"
            b"UczsFMXMWhRzxxDZoJg7rfsVxcx+UMxvRHSEoALiRkpjDcVSWyoRQShGzwLiP85Sz1ORD3tQygiDTMAqaME0Rxj/2Unay3nVp6jU"
            b"Zwnfywnau7nUDCm5lB99bWP5TTTevbxzvqCaZoWN+2zBhRwlWmHp3OYEkW78dDf+YiOfh9PdxF7gpnzW4yjm5bMzhAVdQvI1M4H3"
            b"2Ivm5YapWILrV2crfnKyfU6WpcIXwThoFoxkKhiSuCUxMZRxZRnHhhur4idcIYSdEZIG7kUiA/LI/tE0v5fW1a9l+r2Y0riYSLmR"
            b"o5jrueEDLt7LaZJyuJLLaKQr+EEqo1hl5//qLEUa7JqloKpEPUv+e4pXoTlp82RO43Y6eJ70PZaQONKnX6ygyYnAY8DbVC6idcU/"
            b"GY2XZjQYFTz+oT7rsGHMMZeGcOxdcrDeAHVe++CtjjGuJ0YJFRLHugfD0XHv3h8pQX+5B/DVEkC7zDE/4Ql+D2ptWq6bufpbCkTF"
            b"LJ8/3vu+tggbx+EOv1aM0w1hQFEnRUSaHPUAYat6R2kmIoiJbWLjW+/jHlAWDXOREak1sG7bTiaIt2xaunQoEOhbVWbQumxVruRp"
            b"JXqOgdLbQvRTOuFmC7pP5LB15ZOGSLPi0FqazYFwk89e9tSbP+i1r0avMe2w1NJ6Ak457oSNP1oDSrgYq9s4C1hKtMBWxjmBIRDE"
            b"AVcEx8cuOEV/EiOLZ8xUP8B3qftH+vDTBT/L6ooM00HzuGQm0f10y7UkRjrhEsxLwA9V3qEwQi3OdpngBq6RlX55pSM+zLXks2bg"
            b"mlS48cesBayNlQex4jPU4PRnGDqsiiYz7Rw+a7Mowy40Z7+lep7Qrih43klfLwx1PQk8bZoEt4TH9potnPKVELnfYguDKG0JRLdm"
            b"xUlr4UaO3OWpUyzfv/uTLHlhssRBdLAqxgucJ+F7AS7FUMYHS4OWUmBsCHXYWOo4ISAs8p6JIDmHGNb/ds2Hn1BLfrJHcR7DT1Ml"
            b"Myl52VKBDh18BfuQQoH1rmxwWzzj45yr1s1yBbfIR6pZ+ss6xvmpeR5ubRnBFPjYeHiQ7sihZSXWb+hFjht0LrP2l3HXxhIhGtFl"
            b"3IQH80vPO9BkB6C2muPDyyn25fqf7ECslcobkXrB4Czlw34r+31cufGrQ283Mx2K3Jie+y152OLYY0k2YtIdcnnCo9daz8jzfxoU"
            b"v7xBkQUhEqupRspZED6Q+JDyRKKPgVEeZDCGGstl+g/R2nklrAwuRpnU07fImXRTAn+oTLIPLPeQZTlFLV9M2ojGd15wh7JOEfFv"
            b"iUN7GRAy5L+zVfrLp/H6kPFVfJt6+DTCH2EsB6LJ0PTzTYAkR5PhLie/SkqVnHydr9bJaoS05DFk415ZGRqXvKe2QquyMhWELKoD"
            b"clteTOeLyeSIeyGcZZpOByU094+L5eh8QLbciybKoZJBNxk04ihDMl4Ait6f3qMVaYVP+i1Wszi8TeOsMBPebfPhU1mwJttE0W12"
            b"uJTmZYNK6dAqjdPv7MtDCRm6sVqVH/r3Fkrp5Jw7+EqeA0g7B5B2DqAtrIV+iYTK5WmAI404R9SDYMhL5hUCbkA7mciuHcdMBQbB"
            b"WWadtdoGE9+3xmOhcVxlfJGcyqaYAo9EU+iZbop4ATKlhqlD08mEJmqWGtr3y3Rjrj5k8a2wkBrB8izV9PKIIzmY0Hb4mUulhcM1"
            b"w8rJjftUTVq9JMHCXndg1td5TG64XhoZsvP5JhCFVEcj2Lbn7O9HNq/MiBxTtiwE+sHtwCzJMl3IlrG5p+cC8TFJrNzXq6uyOO7V"
            b"4bZbUKX/+NYxzlt9liWsp12Q31Cqyq282m20DH5zrMtA/NeypdJvr/bSk0bl5MEkVF98+dt1YE4Z/jgBTwNjGsUQm2ntJQXASBvq"
            b"M/c10zHmU/Edq4KjVFgFPgB45Yj0Ev9AXF7Y2/MrW3ayTUEhlrx4McbtWEfaJ1mqkDtQ9gEP01WYTz9FkaU+yXb+kssG1n6hF+hW"
            b"m09Fj1OFn0W5xG/pHmvHHOfCDnEu7BDnwg4VYOZy1avnZVvShK2el307Eqc/AuXCiTeMSk0MI4oQY40N8Xk22FqHpCNeM8edS2IE"
            b"mrrgSDCMeSECcIGC++0S5ju/xY5d14V17qeAL1PHczmlfY2k7xzZsfbGkGfYCieIFjjBL96jC6TFkMTBF+tNgLZhUN3z7EEmzMuZ"
            b"FjQ2DU0dcrFMT8Rlcr48qGKdGJXkcleXVP78tub7X+yfzJDfUomtWMG2z/yiWszsE9wcAQ9Hgx2MZVOU6T+dsnk0r34S4F+eAI+P"
            b"okA0RglaiPgyRhxUxfDaMw6IgUGWSCO9MYCZFZhRZrDGmtkgk+y3/Fbs2ke4lteBWnBFbTTSrxNcR2qBVG0XzZlIN8y1FlUWUyQ1"
            b"Zq5kQmtF4AEwZtIGQ1Q2iF/u1MNNACr7aeCAGXxx66hm+5sq4z6sznbxTwdNhAPzbJdpm5jiDyE16dqy4SpEe5Mc6YYSwVnDOm6C"
            b"YDxobh1xIx2R4rUwwc2rjsxI357BukBFoHWC8H6e6l0nIzfY0ejDLMSwNmjPZHASORdiHMwQ5lhrLqnShCCHpfKBGgeggnRIWCb/"
            b"RNXsw2Lh43wtv8Jv9LSaNVulZMVW8drIlQad6j0Ku71Dq+RryRVAlSxce77JTtT6IJHdOad0mGQxshvRZ3o/d+xGN7Ou92RUnqU3"
            b"kg0ATr6C3uhLSNz+FO1sbhh2HnvwwqL4jzZamPhYIGCEeKKMj2sSjDRoFygEjwWzSiOfegalo1+YJ53Ra/Qg4xeTaNzQ7ysMy4u2"
            b"UjvSM/9UY4lDSdUJ9fDwo5/ScQyxG2/uzwTagQogOFG6ldGLZW1+tAtb07lioQpucGDjRd3m6EhDSdenSL7GZlCVR2MKg8iltUrt"
            b"sdJr4qa2t0IEhx3TaSoBZpyi6zQpG4Ta/nvKV16sP6cGmHHIN0mQcGW4bF/LynwJjQZVS3U0By2UeJPTJs/K+yaTmZYrnUSLL4Sc"
            b"vZQto82UriHisLKuLUKvYMt4X0Y1aOEQcV7jeOJ4Xm2Y085LDkJhI5UOzlJuuGbaAdOSS2kEME+FxAT/7SjvbzQFvowdn2+5Swy8"
            b"4Xg/bHveKf2huYr2TOxvSky/1/vrFLdnUuO3QMzlytI3tKmh7HjsZzIlde67yn5f7PKcuaGfW+77Ofp5wUrHv0VFRcz2mqpCll04"
            b"FQxPbh3byxMW02f7B+OjdKeCVok3eMvCwZt6WAMxbjbYYdchO+w67CpmK1bh6OXq+rtulB8G/F/dQ8iDwxohQgzClmmCLFYaLI+B"
            b"njFWGusS4xJlxDPDCAjPiHASc0lZYCBeKxH7SWZlfMiszF/CrAxLzPqAWRmaFmTxsLb2iFqZnVIrd+0bl7iV2YM60bPUymyuDHWh"
            b"irnLPdwtS92FGMxSD2fggyNmZXaYeugwBPOXD7mKvknN6ZBa+Z31o13uQRASrJI0OiwVPJeOiRBUDHMVBMZDsN4zhyQX0X0hio0w"
            b"xgulKXU0CHDP+6kaD3wilXCeR/h0m9x0Xb9jxpSzE87cWzvmQyBs6+HIRPd1BaNu8tWHY5iH6ft8xzVRQJLx8p1rZPskxZWwdkoh"
            b"cYyFpQ3P5/6WLFjY1NbHMZCmKY4uTdUHw8M5TkCfSx/k/g15u435ABTLTkGx7BQUy05AsV3E2ESX80Bzg2Y9BsV+e+7NpnmhrnLu"
            b"MG++CRgrmEYBVFy5UINkEIGlpjNJqdA2Lpi5xeA8ls54KVQgRHMkPOcIWY8CZp9CZ7FfkMV9oQO+6NaWlELnJ/c54bnk6pYS7RvV"
            b"jiVXN47gobXtWcnVDhs7yy1PJFfxlED6VHJ1zm38UHJ1utsjAdXpThnHVpRaOQix8TuLFcc237GYl3NxxnDfBTFNIBWz+Cz0vWt3"
            b"E8b3kLfHUthHkqrsUFKVHQINjlC2MzzumBt4ORr3+zn8h91qv2AqmOQMBAtABAKEhGNIUKKcTegFrhHjqZEiBugOGA6KCWcVxNlD"
            b"WRSC4AE5qr6Ud6gDJbDHaKuLSeQBLTAhbFuQKnvCtrkkaDONDCRkNxLUD9RD2/XbPfq4Kz16e367bVXan5Q0dT5ccwysoCHkLCdC"
            b"F31TWPu0K7zj6LIWmjdE4aATfSeiCnU0QAgnG8eF6gC/3U6yptv5iOYYs8Bl3MmQYWgJNNR07HkAxXKdAp6iLbrPZbEmjtutymWx"
            b"djf3W3LpZ4a2nRkuQM5OeCwOob0bHO21DHRfAkPreuKOssdtf/NJ9vidHEfXZwKBhbKgRLCKezBUISoY99IyYIgqY5HjiEuMiWdK"
            b"KRbdqUAERw/FLP+D9LXvOOmX8ZneLyNebK24Vuj8/YqSe74M1vBlTHszMsVY4cv4A4uLdzmP4AQiAmUe4N1EwPt+55bWot2CdauZ"
            b"U7b5ZiOuWBqg9y3S3ftH9gN99W8gbPjnVRzjj1w6IPFHm1gtrHIkLhZ8/FVixhC2AZxELOX140QiKUtyiWCpDjxob4T4u4q8fr0s"
            b"67VKYDpS9am0CeU7YoihXpotSe1RqUxya4dcN9hkk3JfBDcR+dJQ11cQs/r1d9B5faLl+Zl65AwK/al65Aug0H8fpVchFdfMO4eN"
            b"REpakIm/zccZWdJgMVFUBkOUYAorHiTyGsW3kaQEMer5n0r0eavReMx8X0bTjVnpTcVEqkGS6nHEvAKMDyLmGr3Ws4sNOkF3Y01W"
            b"NEZl6wJfdODgMXTNw03WifUSFni02BFeqpHvc4Rsr1/5ZF2SD50XuEhWKk5xQKkJz/Vhp4HUCaCUJsQS5MLCmDE2AybD8g20IfLj"
            b"EeVrqCHycy3M6Xq+Ck4HDZwO3genG15eg9P99DO/K7qVcRKQSjpuNENBBiU0lxhxozxG1hDPnQrAbIZee+RkKmMxKxWKYe9TPTLs"
            b"JOg9iHjZ55Ij55mRky4VGFMDxX3VqLGqkUzZQnFZXrLVdjWEZkg3SItkyUtIRtpDrvnvkRx61+PcOd3uuKR0ySBYjyyW7MVTWLt8"
            b"pGxAkWSLDy8caj0ZiMwOqzp6NTNI+z0Z1yaPe6/H70iZCiaygasVb5tO2q3KxMZpCzpp8gsbh+Vk6/uu+2/R+FwKdF/TWrL4MxUD"
            b"CM4R08ZREt0WUQxz7KynkmpPYgSMWFymORK4cg4xH7DShsYVu0AKf62w9kFkvKzlzxby9JwD+SxUVmeq1YfLeL4hgOlYGpNFdGMp"
            b"KK3F5wEYlh1d/P7L2UbbUc26VzlpSBIe0S5Mw/c97UJDsjwas0LAhnoehS07Pgw1XlHlUQC63EnoYD29WGpBf6ElRqcL/xr0aYds"
            b"8RzEuMDn7ji/ExVqWFSoN4Fr2RJOkkpws8lfb1uwRJ20jTpp7wL5chiy6fZtdO/w+9C9l1hvRtvwONZ7IW2DsDQoighQ6QAEIIKR"
            b"F4YZSzH2wbqgFAewmAtitCRKImMx81xgIpn7E2kb5pUfclZD2gtFXCL+7dffbF1TD7ENyzaTyhE7jeLKDVFPqDINi956wpVFXQ11"
            b"mrLmRcBGHadB+Dn/xNKz1nwpvQmB4u6eomm46chq825tHW5fl2ZhuaYaZddGXJwYb8WKmrrP0bL5TGuuvlyd2g9jw6U0pdcQsFCC"
            b"CUmJdpjwgOIywQmtAyUGheA8aCKYi+/FSA9rAcxwixOEy70DwNsFY8uPao/uPYf2PsD1XpBZ2/Xpz4gQ6F5B4SrPw8CyhRdXM64W"
            b"0Va/3lER1F3G6ykEBGmJiRmu+/Tps33zREequ+VHn4Qj03rq+JfgjWisg9GixwCy0Ttnqt58SGB80fpgqyffJ1yLWRZ45kSQx9Da"
            b"YgafbKa4txJOHg83PLR4p8fZeb+HuClaCRy2Eo9saRimYkBT4sPO8tsjad9Dw/A2JTgR4grYxCUw8QGZeF7GLKfgIWgrXHTLWHGn"
            b"ieIEWRas1XEH7jhobZDm/g/LA27B4BIlPVFoRuuSEzeJNTarkcxEM+e5wuuJvWtU6LwS0uClqsGGUkjv/JMdILR58WlPbr7kbBbn"
            b"H9TnUwdEVvqcooWMfDTIn9BnM4AVLXSvun1U7ThKArICBWqSgO0WK0DVJgnYbo2yNv3WXjjnJwl4LQkoeVzIGnBaWm4NpgGYMFJY"
            b"IpXX1CpmglEaa+4Q1VZaY0JcHkMwKO6G4B3EXwesX1/Spzvh3qLnFF9y8GM4Sy9uHx9IY9HqnDbCrZNG1kt8X1PCrJEta2C8Sj8h"
            b"hPBmjxt3OqqFJcPcCQByoMea9PReYzpLh6rlj835LQftUgCL92Ofa5O9xbI1L3LIQ/8mTztr5Wln7VTDcS/0+NC7fWtura9l0Fo8"
            b"nERCC80J106ypAOjpCNGCyG1iXFbdHtJDiZE50dUXE7HOC2GeAo578Fb8tt1up4BdQ4YCx+sCZ9XK5iwAB6CdOCiXOLc+e3SllMU"
            b"+hywPpW63QPW2QD6oVOKx1ucA1OVR7UElFOVxz1qfQAUwb7/doZa3+81UYc8Rq1PcU8T1PozrpqCogTdVGx8oMl4gF1np9h1dopd"
            b"n4ku7tA83ftH9oMoze+wZv8ju1+lArAaiMQmLt2N4EwIC8zEAXkOKmATvUIQAbkgqNKCIBanEbACi7joR69lzLoH+jxHfL5HsP2I"
            b"z2oUc5kSbbGd2NWciUYepFenTDSHY7iIcX2eiYaeJhju9TI9YqI5uiWFiUakOFk0+63LiwmVYzJcuGueUgH7YZ/5DdGWvwhTuXhZ"
            b"zzWOv3UbfacDBkg7Lpiy3ClBsDIaY23ib9UAtSxwA9w47Qk3wllMw5dyDFypvONXysBcq1HPalXTjtQ9rcvi99CO5mDk514aTpP1"
            b"RGqwJjAwlWhKRnDa4DRXh0yqXEiqTgl+P1RaT5r+0gPKgG6syUz2oojVZe9NMxVBxu2jTUem/G5Qj1dHNM+lQHDl4Wnb/kfpyWIV"
            b"fSxVGz/YVEqi2JDGId/VmkkPzx2FXLmrWm1dnUeftDIy0KM8u0QJdEebq9/20KWhx/MF6dyv6P+fYQKajs2ZMOM7u/xPo1yFQBnG"
            b"RYxygzU8rgyBOwhGSyqStIOL7pk5RR011HhKtQ5CYUSY9tyJ8EVdmq8qVw0r6IM+SvmYLeyxbvmBluFO4WYS/l6lBTgSLj+Oki9x"
            b"wFwULj/rMMUVPLOXY7gV99ZgdMEz0IbTlh73DonKo0gwJdt9Ec19GQeWLfNtXP1slyO7x8byRPzb4eBpG/12KHjax74HRC3viH1L"
            b"5E1/fSVt8bdNJqK4XtizqtTQd5eJqFP41iwfv2bI3/SkzvbW6Fch7LCJ8a+hFoGJsS7CHKS0ziImLQ+cK4FE0GCTvoM3XugguQ+U"
            b"Ahfk20W/D3Gn7Ax6+rgXtca1O6/aY963bO8jAcPFHexQ7DLlOlslwCWavQA522NI9yCFpTu3PS3eom4JeJAhnODdqkak6N30kNPP"
            b"+DGON59/eNnlrDmUlmpFe6kmycAnU6NY6cHVIPq71Lq7jnuVBxODWpidYF+/yJZlQFsIfFOxDCmRqgx3cFzV3XXFONqgt2ht3m9f"
            b"F7vEToAXv9tsrWhZ2oJl98I6G+9VA+XvSLBwznN8HR/WezCy+3h4cdZfiJ6dxcOCEq1NjH6Vw5IHrBW3gSAtEUrsVo5bFBSxWiit"
            b"PCI0hDi4EFx8PD37M1hLBjfzXtaS4ed+xlpyxffeZy0ZhMZOWUsIehzM74lN5BFhyb4G+yxtyd1I9VZeQB6Sl8hD8hJ5SF4iD8lL"
            b"OnDD/OV3IS95EIYe5lnfHGzuA0wPFitruaVOaKTAxsDSch+fcYwDki5QF7izMnhBrXfCyjikGJQGYhVyvx3e4arKwFUkwXX4xG00"
            b"w0w56yGp9YWC3UR9Zgo6u0nyjSrzSoL1EsG35XuTRNjvlavvuNNo4Ds5SfZZ1m6oY0p/RZswPhtbMa6M3VRg1sj/tmecDrPsQFMk"
            b"JvheGKeW5vodePHbn0JD5EfxHrFJhjdw2vK9cnqphHbM93qV+XsDru1YTQ7RDoN02LcTkPw7YSA0w4oaCTo+mpIRGTDz2gkUnEca"
            b"mGMsYIFiqEydCt4yJUNI9CeIOiqF/Zt1Nyx4X7Z8epBgbeG+2VZ8thuhR/vWYx72I1zvmkgHlKhB+S7eUTzPcTIB+X55h8NtN3qE"
            b"/z3vbzgmOWGnJCesozXZbd0AiP30N7ToX63itI3jN+yQjct6MIJ4zUE7IMwqFjQRiEjCsSeGMcxVkEIqQ9Oi3wT2tSQnD/pqF5zm"
            b"cWz91ZI5O8qTvfjLJ1R1DtpvRx6W7rYcaKSJVqJsF8XfUyanm8Q44128eqxlXuprheKpAZXNxk7oUuMrMuQLVYxapoFRv0ayYkvk"
            b"4jlZ3eVEvCYblysAjmnlD1QNW6M4xJcn85yz+ZzqOcUgBL6TkyhcAx1/Cm+1DGi1WDIQBA853UPiFrkQt1Rn3W/JlnFlQlpQc7MF"
            b"PDwmbwuG7PeOZz/bifs8YwvJr8mcvUUHQom0ARnPJKLMGMwQc4ZqTLVmNG4qJrTV4KVnHGkaH13JrdcmLsnCH5vqeHkHyJRCYWtV"
            b"2xf9e7WYNstATsV1Ng6IDSqz6bisjAqEYDlPR3S75MMuojmY051zr4MfTpSPnqtohMpOraxycB3slozLeS72S1xiwL3Y2FL7LuJf"
            b"8ZhdoZjV1MSXCpfd0q051peBFiXRZZOL5coqeOGTRaHs0aueY4EepKx/sha/ImthJBKOCss89s4qb4Mj1pogeIz/HTYJpI4ptUIS"
            b"EiN9AZxxS3CIwVNcDtB3kHa/RPH8DkHhU50dn2iDOI7Yd3zgz4mjLxoHVyDNE1xYDqQTpBjw6ouXdQqaaf+mhexBDI1G1bKhp4Us"
            b"bS1819ayLS52iwU0BVbzElx3cOXpKGoMT1ukxL3ujXgV15HClSO75cteWV/ntUJY1Cabul9vtRdeh9Xjwkp6DavHhRWgBlvWGV7S"
            b"NferBNcbDzuDCX+NFPvMpapAFXJBcilAecNDDKCNZNKKuPpVSmmjhcGCcuZD9LJWY62oIoSg6FfVH0Ew9lzn80He4qVkZZswwuIl"
            b"lvbCT4T5mbMC7STWRaOkJZ/iQrvaWJevJuPRiFjxaJ2/3EN5K2pYpKcJKPSID3ZCtpatC2yYwLrfgDWm/cIAKt/aJ1MfNxUcjyHD"
            b"JS3detXXYtN6dcbPIdN+2MaecsOaYxZABALMKQ+GKRP9bQwTCAUSGAqaehwQYB8AM+khkfdgYiTxXLs/R4fxhT0ZNyQdH5CIPSel"
            b"c63bYkOXAOM9ZnAqPtlMCWjzgmLSklZcZjIDgpjaaB4PG+NOy4d8x31WE9hLDqQiwcud43umzApZlukWqx75caJ+w1TeBwjGYp0r"
            b"0I7TYrZv2iMPL7FGqu7e8tkKppiVL+JZQUZCGKOU3PP7dANnDBu4I0/DA7HaMR+GPOXDKL5/8+27xHbzsqGw6HLiS9P0lin5UWL8"
            b"aiVG4wL1SlAPmirnNDAdRDA4yEQU7Il1zgUrvQwGCeeZUEyixBccgooh0TNlz+psXt98Io6bT+BR6/V9WfgO1jwFNU9mmRWl1uO/"
            b"ylwgMeP9kMQuuZ1t0sCSytxOOb7ByS/2eQLLkSjQ7fCsIwEiQ0tKMY1/ReOX2Zo6Gez7XpNZp0nqBEk/r6axY7ldg2XKU+TTAghO"
            b"JlmNXTb9aoOPLJMuhdJeqHr5Czz0+GQzyjHqh7BL5edRZrtnu7Dvprpxk1rpk9NnSXBokuDPprq3EH97r2lJyZ/+vvLrn2rAflPZ"
            b"0niS/C2lNIbtHIwJFJzGyFIZOPJcJj0KFp8+Zm2IPtwYRa1gSOOkro6+tAXwld70uP+4T/4u3GoHh3xqHTD10Efx+q6b7/pyYd/N"
            b"sl4NLDmUSUT6lhXBPlPeBFH7m1Gi23QOjPvuct5OXPt1RK7Ion6maNMoeIjbRYWW8/iUT5q8VVMvnqAey/CoIgrNd4ZG9b1f9aXz"
            b"lR3XIP6mW78Vvj9XjDzy6mf+Hna+/OjVtID5R3j6NlRvW8JPQvV3UnBcDtUtkkYZm/GISaGDBhunABcwQcorTKnS3os4LwiKaDTj"
            b"2EuDZZw2sPbM/oTqJ6H6lbliBydsL+LefHJnGrtM4UQrZynuZdL3GBAQq7tXPTpy6u+zWaqRQqvArpYdyMyFPmCJmogs50pmQUtW"
            b"LI3qgvlmhdORry47zOmV9heTLmIlWXoqYk/fxb02m5pS34jqGmLQjS6J9tz4vHIkrTwd0KDHaY9mhJYXD0+aZgbijn0rzdpR/hPB"
            b"vzKCt0i7YDiHxFZHiZOBYSeY8Z5gR0DJYJxGFpMAUsRPADtEvQ6OY2E9+4GbkGdwJNeRHxn2kTJ6GLUovAkd9SH0G10ivTtRZson"
            b"r0x1jCP0GA54sUqK6/HS3xHYcsDLHA9Jr9NeJ6Q6tKbqkO8Z6iiexJ/QW6LD0EFOxtcv/eSJVz/gkxdWPS210bNKAoYRb5yXxDjq"
            b"ZfxfEAIUhECc09ENU+1jZAyBeqyZU1YyRY36szlC9zVJPAXKvZZK9Ej1c7kd9/K2rKGGk4gOSLq5evAl+qP4aQL48cZoGO2wAxd1"
            b"CIlHDnYao/tZKJtVFAhQMug7yaXIqYbUQ9WO5wSRjhaK7gqoxSbX1shyHfXhGmD+XJQI77mI91Yy40H6YeJKl09w8wluPsHNPniS"
            b"on6cuHhVjfGHIXTwvlxp6a2TKfeQuOpi3CqVCKA4NZpKTQknXlhGnA6MKcIdMTwu14yJAa/+YWi+4VavOvMDRrk99mIvOrou0eVB"
            b"WDsVHW35nFcazUvSo+O9uEJxlA6Sj8W2Y1XwynC0bJMOReTifDvoymierGj6RyyXNmQg9jukJ5RIxo7uRW/O2KKZ+qQfzs/KHcbm"
            b"ueYoHGqOwmEZEQ41R88Ym+mAyV67YOAVbE0/3nj0xkHFADi6X/CAXaIHRcRCsJiqgJURwbDopl1gWoJUKSuBA7YUgaNC4i/SqB9z"
            b"ueeJ3POk6QVSjM9U/fYkbt2Y79UGr1XuDombV7jGqOWRqYkJz2XHgbV+g3g8wfR8xMgsZlT7uS8mDYAiyoaMwHQU2Y7QToFkPfgw"
            b"f8zER25HsDdJPVYuz5bXsyZYadNTSBtSD/rRNZXjXgFKNrx234N0+fuV4X5RsW3xn8LEnx5nHHQQzmpCPQYhAzaOUyJi5EOVYM4G"
            b"SRMHnvU8YHCcMG+QlfC1FCAPsruHqd2Tnhd01vMyaekYC2KHxJ9LM91AlHYhMSxrTYsvBuqgUZnQbCWh1oA2sjW+tAd2x80p0JUV"
            b"qbUXU0Ll/ZQx7d2ha6jLHzfbZCtO91JWojIed9ZpqNkyukO53PTaosj66l626DRQ8+z2RggynJJmwClpBhTTjq1j26pSIbzRgWq2"
            b"oOM/goV+o26tjBqbIX+7RhNrG0/Q2HhCrzRed3lWsaEO05eJ87OazrLPtn6aRGNfv9JEaA9EWCWc8yE+sQy0ZYwlhQ8sPA1GSGWp"
            b"MjSo6AaNBk2UT5EmNuyHOOOHOOOHOOMLiTNu8n8e83TOeTPYISDt7JMlU/vo1ZInGF/1dKA/vBm/gjfDmuj0VeJFAmKQ4yIIwZTy"
            b"TErlEDFcUYadT83gAWksnBDUGgXUUU+1fIbtc/1SZ7HxaWC8i4p32IXzMHfnby/BE9Zw95Ra+VBX6hGr3rRedIMMr8vB4kY1j4zp"
            b"1Vz5kmoBzR6pMPXpkY7YYgCDzbgt1jB3woI357YoP5ldxSybCN4gwGhNBu8z0XOJvbMkyKix91Q0je/RiebO7IY1H68eds6nXzMP"
            b"a3ah9b30YwvD29cnvnjKw/xCxuWXhtBdQ17xtQ8a8s66tx8G16+hHM3jq1528a+eSqSpgdTz7w2PcbVyBgfutA3BB8cVMYy4+C8y"
            b"govokBVNWiTYCH9MovFLcg+Lu7jPuvGUd55xi4589jvsrtxnK15I4TGLutFR1C3WYB6GXMqWFXuOzaPSZOTcKZopVu0WK2m86W5j"
            b"TFFTikS10riLyPOBszFQRtG0Ya/bIVtlb82iM27xujPrdEuKYaeIuvVuvNxXF5nqO70bfElftHkPaMRQoe22ho4qtMEC91uVopS2"
            b"DKV0EJ3qZU9273dMpLMW7O+fB+kE+y7TjL41Q3KK9nWAAifxjCh1SidALxNIR+eNKbOcIRBIOCWCjr/3IBFRgWNrASEVvOPyHWjf"
            b"T/rWi0QVV3VFnoICzyPfYxq4Sfx+kVVo3wbXtphNOviWqBbXRUFPwTxcfK6RZcCxLBkEsmN4PjxX2qOcBzgIdcA8B8sidtcZUhJW"
            b"6a/cTxtkgDqXvBbBkuD+4ZHbyrRJhC2mbTsgnAb4xTJ9d1StK4J+1pgA4zKxBmmJNZ5y8HAPzDansls682iT7tg89zwYP+/RxhNF"
            b"qyOI24xyiTavfmPPPpNifZgiueDxvyBF4qJDcpox7Ak21AiFhaY8OMXBKoiROhcCYtzuFAEQKVmCqWcBU2mIoW9JkRzlR+6TMeEz"
            b"RRQ+U0RZ86o1q4r7fOqSFxdbSy6fKqZsHx8t0nFOCseRlpCVL15wH5QnE4HoGNn2NHGI5nlK8SYCnqcT4vCSVZzAs6OUS2PFoQA1"
            b"QuVoq2ZKu8swp7JiFudD0lPOPSmukkUSi7gKu9+AUSaQO00Yc1EUOBRFgRP2OjgRvIY27B03NrzZt2P+vK+H8qjJ4qV6KCnhEONV"
            b"r3DgIUavHCMfw1UggYYg4pqMURGIT4EsY0xQDshTblx0cpZb9YsTvvd82qNuNjlvcGhd16BAtJLNoMYZ16pul5YoK3/M2qQsgdsi"
            b"T7zmGxCnfQ5me36W4yFeyB3TABfY6ubdeuM8smxJAK98x2JLCY9rcFSumQLUCxebi+P9xecjZkOBCWwFr9Hf7uqZPVfSNjHtMyu4"
            b"wEdkudIFk7tND/2CIh0xmfbEHYcjyaNWlRvumZ42GUNnju5KrcoGkCsbck+5g/du0SinTQa4oYaTAxlcv7XRurVEbU26YZdU6K1/"
            b"U1f7yU63t2WIHQ+UmCRZIhBLTJ9eCq81915acAh7zSXm1FoOEAym2jlDkxK1pETbnwrct6iZXa7+zTK9hziyhktIbtjfRWNrej/z"
            b"YTm9WpK7VIwsxyqlOMC0PermRSdjyczKaQcgWNXdynXIhWGCTYp46pOp3nhNd+tyM+76s4rdWV1ONp/IES0xlUUdsRH7at1PXe7l"
            b"Xjeu3qNPxc6xEBzWQlKFiQERHJJB+uhhk2YUduA8F0YDE9YqpW1gCuzvJ5L9XBWMzDkTLiPPrtLOZ3bLmpxcg7oay5Wj0rGaVSwJ"
            b"Q6jtrcuI9jxZtI4lG5WVLgOi8Fx3e8eWeRGKdzULnY6UKjdoYd8UXXJ3qJAJvEH2eE9isR/tTX6Hckig8d8OwNYwZ85uSbYvN/BL"
            b"wW/5cbud2+Vtcpe32V0+Zh9ayuR9K8f9/O6MPr/P6r4hv/sDgXs2v+sCpQjF3xrR4OK0gLwyinjlvYo+A5Ho+RkTEmlhBDDqtSPK"
            b"GqBOSM7DgwB8Gn6/KPi+AsxoC+ovz5PMsySHTRW7uHJpisMln7I5ZDrDm+XcMUp/ShZCLD109GCHlA5JhBFlMlvVsekOdZ0s1m44"
            b"Mcse88prIfh+vTN2f9BsJpdebZHNDwYYryAZbsrYG4dP3yqXlLHTP3Rjuh+1s+mTOeCb7cdHFTI4lCKBE640OORKW3O+W1/xt5d9"
            b"OsxIVD84BM4XMw9oACssTR3ncIe2u3iFMngcQNEYwoqk3iSIUSgYRWOMCzTGv4JQIYRJL4PWxDlhggKFAGkdg2TyE/f+xL0/ce/X"
            b"qaXea/s4invZSdx71sJ81kgyAxQvPpqt0W77aol723bnn7j318W9nnrEObfGGMoCz1QSPGBMFQ1BM2cttZJJmWaKwJxkiBJhAGvt"
            b"kFP+DbQSFe31CUmpcz2pZwBwA+dZO8QHhApzqEA2yl2BhPCu0W1eectmxSXvHCLaFfYu8R2PfXHtF7JgFEp7du9XYReWZqcaI2tJ"
            b"Rto0sr9Pefh5Co2LKCT3V7KrfmazlMxmq2LJeoYZMiWb5UT8jm9zmJTSiCuJ5pJqvqnmdNs5H7nSI84edhhOF5eNG5eNH2SZ54zB"
            b"d/o/fkedpl+kxrS4V+2Jsx4pybAIzoKx2BpPIGEpNI5xtiNMM+ElCOCOxKWtt4wby73Fgf90fpx2ftwo9s1kjOBayHuqSNTqDMHA"
            b"bCk/2FIo67hzR28tluZsJTYti+30MAIflChVQdKLVHfxNDzuFD/qWUmDXkk5AbcXwJcoBj7VV5KOWvpKGGJDC+DuctNlFrsv7NM7"
            b"ggY/7sabfYIXBez1aPs+vS0wnr231QV/exDwN23v8FYJJrlBVtEkyuSMUI5JJIAxn0gtSYyIPVZUBUI0BxKU5Bpz5rgXSvytFFYP"
            b"AuIRJ3ctup71LzRyRoSceum2saLzSgQ3HRxVCVV8yFYbaM+OQXLnwuMIG1CNjWVHFoFmtwGWjEP82/QAHh05W6UDp4xDI0rKWoL5"
            b"YZ9sWbs1oqMUy6+RNNLXxbOSszkB+nw2PEhSsTaf3R96mqRq2waHoZRRF9NnlVWLnvYdJ1/zHLxNevALdG/VcIvW1w1WIHjQCbKu"
            b"W50mx/CyRuMN6m6u4HFk/yOu+tXiqt4hZLzlAVlCKYmDSDJ+OigkAxBjY3hvGVhDBUhrkEZO4LjapHF9KhDiXzphfFIWBM5kQdgN"
            b"1vkLmh/X9bfJBXpktVYMLtMjr0Scd+mRd/reR8TI+wmwEAdjRvBGZVzLmtNjf5JLeY9xPOVSHs2TVRlsUsRmVc5pKgRQDOTTCZYf"
            b"UuQv99SHsOljUuR3+uNzHxwQV5Jg7yjR8X/CEA8aO88gAFhGguFMMxmi3xXgXbCOY4tC9NDaCvntKOorqvlcHu+QUvmythzbISQG"
            b"77qheXt4c6bWkRW8smp0TK2fomqecytfJJh/H3v8Ee3++m0cw7pbYdbxtmc/mkgGEcAG6FjSRLLyhA7faX7w0vH3GiJlRGqnISJb"
            b"Obwys74ra4KbNj7cFRGP8inHOXG6YK9X94v3+e1BEe+EYa5avpdTeU2+YLr9XssPVGy/0c9z0jda8Mt3OuVURm91v4EoDIIhFZxl"
            b"XjhvQqA6PmocMY+koDy+qZXxaWUX/a13WBGPACcwiUFfKlr6ieT2CXru6STLXrQU3xYtvRw+X2FIOqTm2PHmnzu5GZFGmjnwcq01"
            b"esdTBewppxFqJKHYDEPIF2ZmKNXUzR/iKYSwcHcK2ehOb45/vC+8HpdgJTeOok3nCS1LjE4TW7KyA0XF7zYXQmd6UtmuXPbTxUhC"
            b"469R4GcIjmbEzrIGzbwlyt8RHDW8zrzXLSVdBoV0MOt3kxx996THp4LpN2XAA8VCKEG8pME7FIIOQllPOXfGYOp5cAHHx1QaIRMb"
            b"kvepfzxG0i5wbdE7CI7e2+n9NCpwXBEfVNemmdQ9KnCaPD9ABd5KuLYkGGPC9SEqcJJmvp58/wQqsBWeHnLuJ6jA/WjfgAqcZ95f"
            b"ggr8wQO+Fg/4FUqqT6S0u9Jnl9J+p8bq5ZR2SBQfhCmKNOPMQxCagvPMCaQdcxIQR4og4yRj2iOuKdcGYYkN9ziQP1vx76UB+TX1"
            b"wJuh9ks1BC9KWp1F7NNAF6/pHLowH81zNIVRT9A7Cn/xqqIrJgs7STOSZjHQ7pSMl6gblx3bDNju1iWbKuD6lFRVcvr3EtrHrEnH"
            b"jTa1s4bgJapvt7Irz4yjS2Jl3WgbbDZh67UM2b3EudTZvvuj9vfCzIoG67zmyGpMk7Y68hS4AqKFlsIJF2Nvgr1BBsArCoI6HhAC"
            b"brFVx7DsPhY/CcRPgXznKL4vYwSdnPAy5KQlt4eG3F6eq/VBkySYpLwLU0dSnRqi+8lQk1ny2AIVYtXNtmP2eCKoFlt5kzK5cJdu"
            b"Kw21pnWGnEVyuOlaiUKqi5ppk+vYX3Y2L+dqpKpZ5fio4Rc7lax+BtD3TLrjESdSTTJs2A2+1Qx55Veq4fOwmT+XGxKk3+pyHE27"
            b"+MbClJMtv4Zr6asZP38dr+fiXB2TnHrNTFyBmUCJAyZ08MQ4BVrR6Ge50sIkz8olCKniTyMuDwnzOK6bnyFbYifR70EfzFc1wWxg"
            b"vuyyxoaNlaqTLZ9eZOpk15g6W1TYGVMn9D7ohKmT1VYPcZmos9njSf5NkCv/JmpoUWGEApYWbqRg++4HyCR8bB3cTyHgblHOzXk8"
            b"5SGPpzyJSOUJj+eQ2e03Ns6M75bXvU/j+Sh7+0oaT0DMcosotc7YYJTS8Wfgk04TCU5SLxxJVEZpogwKU+kxz3pOkiMtIeB3ZGw/"
            b"iTTDZ0gzPlvOXogxJwT2sHNfx5TzT0HbLgZrxx0sO4r0kb5+aU8fvBuhlYce8Notx7cGEboL+NITLhUcsM7Ts7C5k4V6pAnVssaL"
            b"GbfcThNqPoo82GL3JDfnq1pBjknjjqELtLrW5RM8aQXp86R0yI4OiIdvxXxxkCl9AxPnu1be8XwyY8a4oXGZEkM/RaR01mmMg0BE"
            b"yYCJUZylt5mSTgTuEFUUURsX5+w3a4h+rIR8RElciXqXFt6Okpj0Wcm6ii4B30rAQ2bdvvmA2TRpZeGei6fOwDeBsIesnAMlZ+vb"
            b"UAHPNDevCeB4+iN6LBoZYM3iI8eCagSijXYqj5xKLvYZzt42mWRBWsKWu6IaD1zi7G6PZFnuInBMRQvTnfVNZ5vsgzv8wRK/Xw5C"
            b"0yN0PcF53GcBhzWoM8yubARN29ewdU10L3ukLt0xDhVf/MsD0sWrNgvw8ozMFuALOOx0AZ5zmNn9olm4+s4FOCAJBitJuCUIiMaa"
            b"pFU4jh4Xg3BSB45kgiMoCImGgiiW4GLSeHAES/7bIQ0upiXPhKf5nH3tAYrgSgL2FnNFykflkcXwC3gvCI1mxOuXBZhnXBPH1fqO"
            b"boK1RPR7megy1PQNRZdI6RzpsIO5ZdM8FV7FR6QRFVuWUJG70U12ydiHDKkoYXI/OQ4deOnryYZX4A9lCMXyWaTB7Ya6Y0G88hlv"
            b"OzA4bTqnj+Twzpv0GhqL5mWBjdVeuZLzPQQdNKF2dfo/sIOvhh0A0oFRqrUhlkktFPYeG6QMt945I7HxBMWPEMPeIhXjbwUqbirB"
            b"kI5Gf7IayfX8wEUO/UfiJhNhp8FVX9Is4Y1+HsNX5olslyN1TCW65qTZiiUQTSJ3nTV202uF/dKJ/h+bsIyOClWHqe90yDLsJD+9"
            b"HL4Z/rzpnOJqH9004oNm1PQS8pUW6ydzH/elnmge1UZ/Lxv4gWyi6/71nMuTfrR+vu/P65g9Dzg+l4j8KvDgR4vkPis+IIMNtlYT"
            b"q4xJeQ8NmDGPPSOWasGx4h5LEcNzxFIjR3yEDfEixFjeRXf1VVJ3/HMUouf8oc9L3fGB3XNwJKtM0WP91PyczOiSB9kjteAF+M4H"
            b"Z8WjTupOLeEluyB1d8yWMZtsxkTtPGV9dERUDdsK2jKMzuUvJbTnmeQZf7/WXVsi65TuaFMga9uQ92p2b9C4eylNZuvHjlRAWt3v"
            b"S6SXr62PGQ8iUG9MAG908mXAo3PjMW4MoJTRHIJjiFNsidAyIBqiQ6Miejtj9Z8cWD6puTQNfw4yrI87LuYNZpdD3tcrOPVZ4Da5"
            b"eqXCRc/D20mF64DyMpsUqSXMN1TtFPywCCYVUCpVCwJCHcE0shGhX1M8O1ZPqqnf5pOrDGtNrnhSPBte7doR7vBd/oSPT4SPgAA8"
            b"MKyoB0SUJkjJoK2U0eWG6AGMRsEApzT+ImJ8Epf3xCthfQwvnTH2S5uCH/UWnPA0vIikoYpTsIZITO25K+NOTLUVolG7okf0T7gJ"
            b"0v4JHb9QiolSX1+9W69eMSOEGFGfmXIsHkqpoV5fI9EdoWeBh7FCxqY6fpuhFJdsWurMhmXnUblySnDB8NIgzBTfj3YnYK+K4RJk"
            b"3uRMgBuQVDj0dlAZfRtsVSe3MS9/4abkhfflr2GpPRLWfHduhIrUH1KfFzkQnu+aHTtlo4tjDFnLLKaMBW4Dlt4lUq/oy1yQKDBg"
            b"wnDmMJjAMETXFhRn1GIdV9DkDwOQludlHfo2zm0ZGN2WbPSI60y5D07IR663lCmPljn4xPxaEMuWQBEwOY+lLsWn+TCFO2ZBTR0F"
            b"crySHyg8Ql13uIdsVBWQVAWjHlSmshlGm3skx8Z5jMn2SUgpvS0J1Jfot5Xy/JPyGhpP9viTrTPp+NUPgvTaChmQ8nE94CSXPkhN"
            b"nTWCEhRXzA4rHpCzBLu4YA7KII2jf7MgsMBeYIJp4F/kychSAT4Kt06bOOWcCaVAMtvpakrPcrKSyjTcg7dq/Rvc8FlDN+PUZ4nm"
            b"wOQMmkn6+zJbpZPZ3eHl89a3rMlFeeInFxGh3igeJRkogYcrGBGr6RpVAc+zJ3oqby5I567o7JMjV3T2yazXfXz1bRom826PnJTc"
            b"WB4eN0a+1kkJEqgGhwPHGEFIfelAEfHxtJx4ygVHlBgsJXMaIRrXnp4bi1WMvCQJb4BktqiUl/L7Ha8bh0bDghrBvO+hHsElhZUE"
            b"SbSBS9A21zTSh8kEI8VbFo7RDOX+sMQw0kDLKxvSnvs048qr94KMf1RP0ZjOoq05L+kEwrnd0iYcSgdTZTm94iHJYJc+T2YFl1m/"
            b"gt4itTBi+fQi8WYdljYN3nRo/J7LwDxq9L6Hk2xerrRMvyOV3hcS5jXeSyEHPnAf4yznpXMx2EfaeQtADGMgqbWcaC4YWKQTAUeI"
            b"URfFlGL4kYA5lYC5Qm33HNHTXEnmjpBKerYrYVJD/H+kK7Pn+qhd+lNBmyntRpMnG3e4L9KSDlxEWiijaNKTMwokJqtctGCkElBV"
            b"tz+zTrekGDZViwMqxHnJ4m6/+F3qjYUMDzpqPGj4pOdQR1ip8WjHjLdhYI7AjtD2RXbM0t37DS0etLR4X6WE+/cThwGQzHCLALRl"
            b"lvEgpSUWkHWEIEJZsIhhbrxx0hAao1PksZVS8IAFk47/idR4D1qA8K6pJ5lS1GQBlwTcaExRccdXxREXecT4V/T6KsdqiuKjiC+q"
            b"bYc6xxzvo2pJNj56k2bJg/2ydfqlQeF4XvZpeOUO94QcpsV/1QZ1aUGFh/up2kqAFUOk25f3N5480fu1I1vFp2SrpBS90wOwPpTD"
            b"MPZEVnxpg0isC3i+V98Hlq+0WD8Jd78pNfAcyBFaTpBmAxbhmJWpr9064mBq0EG7eaBtZaI7WNEyn/yg3b8c7Q6aS6yd9RQIAgVM"
            b"CwQCHCS9MeucVLk+boL2wL2VylKhlYv/UcER/IaU63Fx/IgKaqtzP8q9PsN7dyn7eplrb8x7JpPp4A5C8CnX3qUM6BFAaaZbsAco"
            b"rdlevIv7K44IyyYqr1hRsYMS4aKe2zXWT7O6R7CjedL2GHa0Zb5PY/ibWV55F3Y078wvxSjcFKbwQ9jRWWZ4JkTQyw/QaXf/75n7"
            b"/SRd3vswR0YRBsFzaYznTGlPFBECeRG3mUMqhuTUqxBc/Dl4Ex2w1Mpyba0Lwrs/jLhUPULJzMlGe2HbbU3Sp4FviA50mrtkprqb"
            b"PsUbtmmFDzzS0m0BQXAJcj740XTKQpNCqGgQUQvSoE+kJ5s0JRGhWhc6pak+Us+FlQx1p6wIuCIFxAZOKr9bMTMXvM5oaFen24vz"
            b"5jEX097z3u3rv4ealw1RSvsaDl3zmdwW/ShHXD6Rp3ylWyP/FgfvW4l+uQN+0NLfhrdtuuREFrEmY+i0pT8P+n20KWCxCQyBRzgQ"
            b"RFH8hyWpXLDScstRDHctRQksFZdqLlArmPQiaJMUExn7c+RzP9XrP3Wk19R2r+htfbbVH5qs8ZVW/+HLOWv1n7IC7Fv9RWXqw402"
            b"4qNe/zYrfr3Xf7fXlV7/PDxywA/wxc3+N5FeOAZJ6du+Q6Z6lAKXJ93+8iQBLk+6/TsK1eZl0+0vt25/uc4HckW+tu897vb/kc19"
            b"U/rDCiw5SfAyzqQmHLiUjDEqQeuADHVKCeOcSAOUhmIUH2uJhKXaGaq/CMyx4TXgEWSDnqE2HnIH1swBPcdqVYiCGBMWU8ADHZH/"
            b"z+gWzJUYL0kCJIN4OgFjUL8zFLAKuqC+v+FQh3EI7zuw3XATi2jXKFi5fafjd5PHACX9gcUgyLhOkbu9cA6kZS5kApt8GYMgY7Th"
            b"8mlBxrth+BHK44jVcGlr7RtW28Q2NIlt2OvhNpH2/OUWlD/SefkmTQclQ1JdXVk3HYfob8eXnBJrxfAbNHVeMmKl8jqFQ44LRbHy"
            b"znpQVEdvy5UhFCVBrxyYM66Qko4j+oUNr7WC9YkGfvGoh/9KFmTsWb/Ug38ZLDLhBhADPcBDupVzjpNdzz1qktiyRJmyF5UVp71e"
            b"vLn/k5vYd5rhpjVjn3uWBQ4il9aPWo3GB+n3fIXJOEbVbOXxLle5DmmyW7LOjWhI1t3EuuMECJPNyt3p4SQ3WQTK0u0ukUpT0Gs2"
            b"oIS5je74SILI20Ijbx0yb0uNvCdU2XIf46veCb+QTuXXMQw8hI98EffALD2dYMg0PnRJTEthpQwDxgJRxjBMqTacgdDxlXAxMmZK"
            b"SwKJg4ApC0SEH5nyhxlqdF/O/DyrfD+BflEk/Dw/vO9GPUqj9/NM0aSqD4SsKrvTyy5nTVRVWCq8RLmqAaLw3RjS2cssRLDaSpl8"
            b"uQvD/U9GWTaXKpidYD+vZMsyoGcD42fYCis6olMo2HgH54BrdqquxU5A160UYsNAyMuT0zISLAXDqzwFP8rlt9PVgQUlkWZeQfxj"
            b"vIsOOS4YpSUsBEDOAVgbH0se3yQovht9jxbIOANMP+p7m4bJL2KF+dVQ7CMgNq/uWfAZLBp6Px6NmEQjzcrU+pC++3GAPw/Kc1og"
            b"O7OKXBxJsYdpIrEmZACjkLJv9+2zu7RYEIk3rcbqgtm+4Sb59EX0dq6hmw+ES1scbbIRFfbcc5LTnhHrJjT6JvTtjBKreNANzrxt"
            b"yBYGLXtMdJ6K6aZCLveY6BlL4DdFPXc52zZs7elaGtD9JXTzglZGjaODBu98hHpuXeKGaA464BAcOGWw8AhHRxjPGbAPnNnUiWIQ"
            b"N9gSr5n1wlODAJGgWcavhecFBtkXMvftIcpbf8FD8Ot06f9gtdydHC1upiBYxcK/t2Je98IHyRSANjBpvl71MOBslaoqwJZantiS"
            b"/x2qLVukv3JLsVY5Fv5YTmHEfV+SQNjzk7MdNWI+RBkWMI7YxoAj1+G1J842ouROBZkpbfeJ1lI8/NQiH+ITkSarW10jrSzguFk8"
            b"3Qb77bdqTpU2WVna9Ju0JbSGOLD4x+KNhq0x19oghS+RJXz1gv+oX6QtcB3pC9Y5+fpy/21Z2LhuZ4F57nxc2guuCdVa0LjUJ9Rq"
            b"Q5gViBiNlUdee0ws9TSRq2qu49LM/OYSMhfxD1fgChcbEg7wEeJp5AOtCAW8NbEtalenaYb1Vu5Udcr6XKBlfV7j2MMd8sknEIlh"
            b"CLtWDGg6JHZI4HTlpRuDrTMMKb52G8mwV0ZgZPO4oIfV6a6J2aP9inWWov2kmsw9B1w1CJs+inZLrmv+QZWwFzOUi2/eNmRx3LL1"
            b"23Ld70jGdY9WGFEJP8Iyr/W8hhHg1kUn5JVNyoeeIqxNjG85k9ZZp1CSRsTcGK0Iow67+CMm0WFzTtEPdfWT1NW0+sLk4eTSRXcU"
            b"GV+qsfHihOXW/bUyV48XGU8qM7bqb8FeHe/zvU6I+VqdHdJXs0P6anZIX90xrK6eb/buD331dd4bggKTitAYQRrLKGde8tTqED0a"
            b"tophJ0LQyBNmiOJGM2E1lToGnNYojcJvwaT6iTYH8UVMqgP18wmTKn9UXWczQdljJlW0dsf+MibVwS2fManux7lnUr3X8/XDpPrp"
            b"9q5LTKpNV8rjJq6XMqkSEJwar4x0LiCudYzOUtxmKeMmcKbAmfgUUUUZUBWw8EkNSgTpkkjon8akuq1kc0g2LmLXeIwtnx71xjbh"
            b"WO0lPVypXeI8TUfKXH5ErsicFr4+UA0ms5RjqhEZqwk9cbDeznbpyUcgG4wTLK2n5L76djpSNmgDsiI+uNNzzfEYbE+A6LMZ+cf9"
            b"HHsqS9238sbi9Shak4fRWl3SNjD5fgt3S1w8X/7yycYSuclXNEz9nShVCaXOEiuZIBJbE5hwiogkIc2ihwtUKOFwEFKIuP6UBnFA"
            b"GrRigQgbzZ+vrNDjfv216nte8n0ijXekQ/pOEdLDxOCI05zk8fBOT/SopDNbTs/ScviuQim/SqYyqygVULtaGohaVpRB1Kkm/JJt"
            b"Kp10PUQ1uBsqyWnAxfRTRFv3kJHysOghD4se8hCdIw/pEO9m6fhm8H6NkHdSah0m6i4Rar0pUUekkJTGpatA0ikehGAcI6Tj/7EA"
            b"KgPhSkhlbXSfzASwylmsYtDHsMH0dyuRiIcKR2e+9aSu8cIqykk5Y8ehPe/zvNTkST7TknrcD0p2t+b9tZGlLoIm/nWY3yae9WYo"
            b"eY+9cHGqaxWE9kUQ2gaBwwZtmvDbAghtSh70E+WPq471d2zA/EVtlotTtUZ7y7lRNqgUZnqA6GGVjRGmd1oixGgQYL1AxjiOOXE+"
            b"GBWCtpQL99uxFF5ttb/oAect+VPpuUlv/Lm88VMa9NcC57Ej8kTVmODN25bkYXVX06pGaWrMltHVIblvQt8TXhXD9Ff1VIxHWibl"
            b"dqS/cnvM6PltOVpzDGv7PIhyZE4FEzN80ThnJDNIuCrBV3u+kjz2ptEifTNP0g3epaOtjICctvyAvAFJtt0//VbOaa/Bc7dxngHF"
            b"U/7Zlpd2IyCc65e+uLvoh3HwmZZ7ykhiHadeee4tpU5rJHGMv5G31jLDVZDCBggSE86AAwFJObWWJp1n/Y554ZNo9E+RrMxR4VfY"
            b"qpreyuo9+dqtvhebX7i/0d4xk5HTDxXfTDBejrseFU+EokU5LmDVSNnw/hQwEoz33VMNAe0Dsq1mntoxXS2zIG+oq/Dadtnb36FO"
            b"v0tXXkaQjRko4CNQf7zj8b4VO0YqrKmVstjPhcWsfEGdw7+bHAH+EkWt41oXbrbw8Mlxyz80dawGYD+8tzFr9a9eBib9fpmSxr/T"
            b"FrZPf2EOZerihVYB27gESDxcPDpzpLHUiApjfXT9muMkoKOCD4BBS8YcY5QZRSnz6G8munpF33ToqVQHZlclSOUapy9F+8YB78Pi"
            b"bBf/0FbyBg7Ms13y7pfkTXNnZjIkgsmFb5Y3eayxN1Nk0EQq4gNtdDY2+6GgBqleBmxUo1U3FWvzMSR6VmAnz/q3kFK8Lb7xDisl"
            b"RyaqdaOgpZrkSrPBTgi/WRtTsy2mHhpCa8LkR5j1GpyAaiDYKcAo+kNvMPEIeYJZdIEipHqc0lLFr98zYZQIykEMYLDxTsdAmPzm"
            b"aeUXJoMPmi7HPPaef6ob8yoqWBOxi4pMR6mEJyiInIhNUIM29S0+kUq/yFyVzpnmhmOg/g7mfit9zVW5C0Qh1SXGycJ2Nb0h2TwN"
            b"iiKhmpOQY0hFtqwsgk9mnCWRkiN5ixDwU0iEwwSGXPpDoe0PhT0t4Jqx6F42BgtMS9YuqJ/88yvzz9QKYqIXsshyp6OrSYll4oAz"
            b"zHXAcb0oKdbYCFBWGyMDQYxobaUxwlP6o3R2qnR2mXVq57zZcdP8RWxGn8mYpzFuKaPV/MUbhNGSfm0aAcZUbpH2eheO9c7S4fNO"
            b"f4jeWfpar/tuWuPYpboHTXXvDJ4xI/k+O9pGxrokGNr3FkDtnrT1R9Ps1ZpmNGBFFOeGWO1kMFo6b7WWwClE1504AUUMp41TmHAu"
            b"WPTbjEeXjgIHDv4LuQLpVxcWr4FMjx3yTglhrgMzHO8k6N6tnjv+kC7t23l3UeaVhiCglS1AowhNRkEQscqbNXTTPcGTSLINmxxC"
            b"FTWbmE5ZYer0uHfci6enJfvcErPMpW0KlUySNxizKpOx0GrYCSfML28nmXCztkdvEwPiJrGLG1pAPKR8N1LAmX9drGiTGN5LJWyp"
            b"3bbE15fyvmvR7hb4d8+50snVPCrOvY0PkCGVOP+4ERgpRDFjPkRv7Bxj3BngXFtwBmnPUAyjNXWYciIwcIxiMA3fnH2qDWhepUe5"
            b"dqrWcj/ftUYU3ykbG3HZux+4bbJkLjAsAuro1CWR2oMmZcfvT3dNZsmi8cqfcpzpOMlMStTQwBwNMF6BLEnVJfmKZqyzpS1ioJ46"
            b"J526IzVAb/aw4l1j1tJ12iuhbyR9nehut8EOpRtZwzXVkvU10l5fwzX1Cn9Y/d6sw+ux33sh0VR0YSIR52kXPHaSOk2JE54z4Tyz"
            b"1GjBo+Oz1ArjicaaGO40cAECgiHqS7lPP6lueJVG/0gE4BIebdqWeaLVUi/lAh6t6Q2dCwRek5g5Y+i/ikfbdFEG+EUGo20wiVqZ"
            b"m5rPsGVNo+yBBmJVBatSjFBTxWcXISvUmS/3pGvIHWRmiiFDgGmbMZgPJo+i2O5hCtO07xSmcLPp9geY9k1bdjsdgqNM8A2psPMm"
            b"3y8ALTBmbPTzyJNEmaUJIKuFwYJZsNg7RTRQ7f5/9q4rvXFdSW/oPiAU0mLmAXH/SxgEBgAEKJKW3O4+/uZOH9EuUiAlFwqFP1gq"
            b"JcXSaEOsCCJmz2BJPPwaLo2NLQjW+naTl7jd/sUvnQleIdnGXV2+N0LpPhXQde7oV9M5KNfEBCQ64I1xjyh73Qb+gmrKDFycehOY"
            b"tO5WB3XTHBH/wR2guAuE5U7TvzCAEnfh+YL57enBAn4QS/OjoXLlCTfW74d4mdWs0ziSDCGrfd/HUOUc9UVDgrt5e45pWDNp1eit"
            b"DAnOTMxlladlk7Or3Fy9rL3Md9jDIVH/KN2YSf/3M5YEHxPAZtpxQk38YwUB3hBjaUjbchwhxGn8f4dpkAZr75iKlQsnklulOCCN"
            b"PEG/AtiH/TkKZfGvKlUWuXy/Ov3rGjaxdSsHBip/WPo6ZbwiSys6QO1RFleUCeSFSHYGOouskyWqmnwxSmjxCyXpfFz7uvgGvEmj"
            b"YaZ+LU/Ur+WJ+nUnyjCGOvzqXz9oS3BQWjpuaBCgvPNGYhOot1pZ5WO9aQOJi0dEhQatubWCayUIAyqDt/8uV+5RIfgOOt1cBWHQ"
            b"xSx7RSkxbU9ll0EgowR1D6hVoTHkXhIedXbwCrGK/7K9jJxdNg+3PApGoOEN8grKRfo+xlDFgh6FLPL6mZRRt7XmbPx53OnSePUX"
            b"W2BvFTh5osGRT0hCQxRjtdesL2+nxJf7f0idu6/Kfa1HMdPszo2Jrcptj840uy/0KapCt25Z1NGHwviXRPcnSHScOceIR7EIxpxj"
            b"6UQsjeNX2TgnkYyVMzMSFGDLVSJZEK6JB8+IZl5r/W3SstD3tO/NKudTynNp2V7Lv5OWhVWO5lzLbA1+q5iZrMxmr6iZbTo+cFnO"
            b"rD6lm3BZCYuT6tbelUM+4UXds/QHteiePcis90kTz+RlpzBhdgoT7kgT7cF7pWb/HGjhG6EJa2rTMYExZULMa5oyCbEEFsEaQbnz"
            b"TCLrkRKWMGvSSBCPSY+BCF5qIQxiPwq3e4ZKgPN+wFnqa+Fcpb5pgatnYF7c9T5hoZ+RLQTzobk0z7lDZLGDvaITo9grlXub+HBF"
            b"Khte8ipQAjZ8Gx8AdLtgVVAVLZ0MN8a+LQmsAPWXfgHefbUaRy1RIh7iETjJbjq3uGKU1GQxWqMSKKm7qe0RK6G4+u2IMda3FI75"
            b"jpJibj41Efz5NeJzhMJbSWPcCBsCRVZaS7gLTntutWdaKC4MSJCCKi1BSMuxCUHzwA0Iy2mQTH+CNFav7Y4kWPzKfRXOqLDd33jb"
            b"YyzNQHolEywptf1zLeqE8Wskl4bv1t3EK7FJnRPImvFfhh1cRV5co4GlgKTzWNXm6x9OFyjw2laIC+lGVnyvKcl1eMThaaaLpoeJ"
            b"KSGi6oPu9LlB1s6xRbyH08LD7Z5uO8ekGC4fG6/mr8UtklhF3qpfz/QJVvFbqGpJ2OgEe2UJnSzuC1pY10GF6tWPp9s2UIDCZ5lD"
            b"AT7utHpKCuM2cJk8ruNMjiTGkjFQsXi0QEkIODHEZBDKKKKtoUgzRpigkHRLmcEPnFbZVJVgjgdj74e64heNWjZDp+4La9YO+agR"
            b"PtpKrpCwbPEcvAGEHaLLDkBYtkjPim/HweJ7OFh8HQe7PWwx92u5yaD9l3GuP0Xz+8TI4LXq9zv3k5yj3HprJJHCg3MxsQWPwIPy"
            b"jFtHTbDOMmIxiSnQaJAhZjitADPltfsvQvqb/fQBQmi8n44HvJ7hfjo7LCavE6wuMwRO9tOPWedkP/0AjJ3vp6Or++n45X76uwD9"
            b"v3vo/z1cPw/YxmVyIEKqAGkjzxipOcck2EQcZRQ7qlz8kSJGx7QIjDifFaeCkvI/pjj1MXMq1q+Vp+ZUOR++MKc66lvNzanYYkt9"
            b"AYz0wxWxjjZVN1fAjN9i2o+l+5Z8tJVyvWXVCAsvpzZVc6Y97NjMX12pqy3C4AmmTjhjgMTvN4AN0msqmSUOIU6IE8wK5UOQSFqk"
            b"ghI0GfN5HBD7p7nyXyiqvkSq3+rHFwidS9T7S/vGIyWS2S5HiSk4JSCy0kadk9nLLjOB9eLT4jDdTInDjHG2W2apht9/BGSV9qeE"
            b"C0rhkOPkvoO96dFOL5/Hks54yJon99bKuBcwrTaeR6KpZ78h1RVIc7U1d7JN6HT+CqpXv4z5tzLmBdZMgqDMcaOJBq4MZZI4HFOw"
            b"EBZLnNiksdpUCJAmIniDMFc4ZhODLP9rpKU+udDm/5sttS/sHd8QO+l2xwer0asQ0lfaT8OkWzCeaGR5eoCtTzfiyXAGy9fNkk8U"
            b"8UbyaUrLvYNrzeu61FKIi6fq+ZF1RCMkbIwsg4k3UShSC43p7D5yaLrrh5maUsYA6K0V/8mO+Wr0xaEqXbcDunQ1qw1zXguuwhKx"
            b"9jnro/nav/l5tZe+KgP+1L7AbbzlF7L5x6SnhOY8KKdM4nEyq4WmymBhrNBO2vjSQPwWCcdRrLkVtQECN8Lw+FMPJPx14PvL+e4a"
            b"SB8WbH36Vx7TIn0EZc/XyrVbu2Le7KI7YwVVpow4ySLZSv9vDLctOIWkOSZO4xyOaZPS51aQ9xD4JSyn/aVF0hA/J4j1HFvGThRD"
            b"tEG7t7yD5lb4Sn+gRFLSAW13gPPWjZFLQybtfR5ND8ofdXtGiiszTPxTIWMqRPt1yOMv0Q/R93e6HU+VqsYQAXzialZbcLdm3GOk"
            b"PQzOwTX4/hdj//0Ye+E9sgyhpDbIBNWBxoKLIIaBeWNA4WBFjNDBBxkYk0gKTbyJmdVz4uAPt1m+SDC91h8Zdyr4S7LXzPXmWnuk"
            b"77VUrMC+L8E38ZSzRHyZ2nqZLMtlpY+w4lBh1XzlQ4UEvnZoDi2Xw8oihZTKn/B9bTGcL/L9l7hK61XuWq/0E3Kv36A5OLOcmWfy"
            b"TmnwJ2kOvoPPei3P3lV8/VwTJThjAlXWYS5I/J9XRjBsLMbSyViKI40lQ95xSanF2jHBtQ4GrKHEuW8CulZtkunG3KlvAlYnsl1d"
            b"+nvtZHtw4T0SL8t51YV2Z0J8CL4iorW45x5tbtnZduCSjUTnOxsjJKeVuBYbRR73AskRsDWeoQ4Y17UmJZQc3ra7ZBpZjtu23naB"
            b"7htWBfc7x6TqHJOqP0w6Iy2y9XTJltX212y6/9YQksYvl0T4x3Wn1nKyEjYpe3CvIKUla5bNuuM+3PvkS5b0Fcs9Ty1Iq4gxPoEM"
            b"nKBKSWGlQoyD8nFBRAIFgrX3XBGCiNUEWWsUJf5blQTfksReyQ+q23+bvQng3ik5sxCgK0Hz+LajSnOMIZ34K6LO15DsEIgdd38U"
            b"MTlwDNJbkiFba6hItRQFpINOFFUqCqJCTyyR7SBYyb5Mot5OcfncuvAsTFWqU9lWs4sUoTiuB3JkQf8jytsRkYFqdRx1idyR/PeU"
            b"/+I73kmkM/l/tqbBrTzcU2yhx6/24vvr8hu5FZH76xqgOtpuW5fw74avfoekX70DVxeZS1ts1LP9pHDf6dpcUqGRYU4YFkTgDmOt"
            b"tdQWaWkUKMYCITjIWFeqgEnC95OYg6UPRgiH6LWe7UnD9jtI7d8vkjL1sj6tUvGBb79UjII2tlLkUH3eNKxKC3pUF6JsGydpB9nV"
            b"j3zO0T/dFDvAIrIlQV6lMwBUDZicuHLl0GzfyDhq5AX7YadbK0EPt8Lyt+WOEMm8gbmwRKuEyqslNlR5c39dMqisGqyyKWBhExw5"
            b"COt1aqn/tMDIH2pxrqlTEMQQcCyYIDFTcomEZFYYy7TBMT3GFTnyVjmnGcFKWqGt4UFrBtpy/lu4Vh3Iok+kRn20Cyvr63VtmQsA"
            b"I9Q2MZaJ+UF9eZCW3nutXWQOKXqjTOwOrdv8MDgjBcbHswuUqja4e1SpcZASlySM112Tg1FXfgA5qm6rolXkSR0/znFbdR8MHQil"
            b"KvK4ZqW3UGMziulZVwCqI+j6BfOuAHmJEFvj6jL2t2B9Y8GqieLYKq8EBuqVFtZpFf+s4trNa4cNp95YDN4JEjhHCmvmpKY8m7Ko"
            b"J5tJ7CQZn2bidzsR3E6q892Jyyv8Q27lk8H1eXpsfjDZmZJ9qhzuOYlhkoSSl4q7q1zHiKvGQS+aOMaf0WGDI8dkBO+eKOncgWGY"
            b"5OUpKDeNfMmvhBHenrTdRo/kTZcv8d+TY+eL/9muE5vuLZ39plruVy9nWbZtv/6YPFuI/C/AuW/KtB/bV5I6BOEFUkZy4qnmynOK"
            b"AkAQ0ngRQvBcBQJcU8aE9I4GpzQH6RBiAP8yO+IGneEomNcz+y8pwB11+Tprp0tchxfL+X62KYvtWqLv4JrdnsK29C52+kLTRKUX"
            b"vQ/QKLnOJP5Ep/I3EIktvgmUr/osZPfZXtLs8cT0NuUcTCninejf8JbynZfoh52HB6RbyMPa06+sIFayMxfcjVw3LG5zUIhsHKp9"
            b"/8qwezdy3clna7Hb09A4VKDfX+LEm3Nzxs0KbC1CSjkWkzVCSFihUzom3FiN07Y/AZusVcA6j4hx4IBQrcIH9vwXDbs5E3dOjaCn"
            b"WisnKiszfsLi7qSOfPmjMlW/Yx6vEQMkW6/EKo0l3IYeuxdHu6oZj25awOaGwJZoaSd1wHIYYdvw1v4BXRPhI4HRdMGSuSkRdZdl"
            b"uey9dky+SNYHfyaw8t7C9Ah6YkvUvitFKnEVXClPfffG/1vFVS4s9htx0lfCKu/d9Y8fZvzCaYKJMkZLZUFqxjHhmCnCScxdIIWT"
            b"TsQEJ72MpSb1nHAXF/maot/m6VMBvfEq/0oj9nJLdFDX1baFg9Zi72nVexY+6reOrKN6p8AWDCpzjfS6wTnAjTaWfy9Ro8szRpXk"
            b"4ww0enNRfw8uOibO4ilx9uw3s1Yr3pqlNVx0/9noVYn7bZu+r22qqAtKKjDGWBOTKaAgEdEoeGWw1iZgKeM/3IVgjLMWnBGWM2Ks"
            b"d0zCj8u36JVa6h0Zl2v5dqirQi71BoZKKTOkZ6f9UmGM9uUrbhlB/LC/zstAU4dQVrp5e1odMgy69iyeNmhn8Nl2G/7KDLUMMf5b"
            b"YKVFI6dJ1OOhogUbUWnvyIlFa76fFEklwydv0vZX8TKopwqq6YtwD3hVS6CSStdvIazWCjCNwsFRQeYMkrUhBJqXO+Cq7bW2kT9d"
            b"WeZaSi6r/m+QUD1PyaCpR8LYWPSKQLlIBiPKYexAIOoUtsY6aokOjiqjwSthQEBg1jMb8F8htfW+mripUNWLdfYJBKBfbe+xcEl8"
            b"vwf19+L7bN9HH8FWL2qzHrX30XyPaqi9z2qZ1BfS+6UAgk43MAvvC7V/wu3KQqhFefCJG3V6DndyI1RqgtAYkshGE1A2WXPsWsqm"
            b"TFS2A6kOuZHvKfFvcJWe66e+LkHfqLClOAvIYU81ExgZhT22WhPJlY7LPual9AZbrDxOcqoUSHKDJpqyWKhi0P+qH98NE70plPR1"
            b"W64Dr24udXHx2rjUTbdbJghVehGkeqRUvUCe0oGF8h1ga7mx9AmBogAHe+yh7lYJzTmZYarI+KzG77TE5Vx+w/GPrv6A8V9YJbSP"
            b"Zw20DWA5hYNgpMe9Hh9aerwl9KEWwC30qzxBv8oTW2k5FbKWC08AKgQt1GKKB9mX8cvjLtT6U1xJKf4KAny7IICSAlkmg+dSMmmC"
            b"BBKTPTIIYl4kXMcSF1GVkLLCGsoMow47EkBz7jVT31r5wo+yEbhSkT5XpbnQG1YjobEr4mCHjbATB4OTBvJ4poJ9h/+AZp1BCRYy"
            b"LtuRuLu5Szd2unEbCgpgUZZZZcv7J5sumkOPuN2J9GMBzGYIGCeFpqYOK5Oj3FemhlVIg5v7bJgLIdKMeb1lfK7ahU9Vu/CpWhhe"
            b"/bUWIEJ7hGsxr4bvUH7Od0ZEy3hYjF45fLVq/46tuRO5r3lL+fW23YfkvpTGzvNU0iBCDIt1PBCI/yqjAjcSAZBYJjFIPF0GzngJ"
            b"iHHFDTU+fnn/Cd2BlxiE+l1P3bEuYRCOBP8rG3czM7BzDEKJIwe4whyDADOlrYvbmRMMAqzNhS9hEG42KNLl72GwRhgEOcUgyCkG"
            b"QU4xCHJvyM5e/hTxgQc7Z9+4P7YmMCu1dNIrojD34LCxSGjBATsWfKxFEWMQtCQhCOqYjPNlLFEtVdwprj6EQfhK1+KDVeUQ6zTE"
            b"CCwMJCTbAnmn7N2sU8e9kokaQaF3ISXa7auW/LT0K8i+bz9ubBworfhU57XwWfPOwj55sEa076BUS7amzN5I6HbpjvlU7n0EAkut"
            b"WAlFjk9K91tOYBQXLcgNs0Dnp+VnmT/NL2kx8qmsVgkrY9raFR925Jo7aEG17wbdHtyskcEKRrbqgNRHTfe4ayQv7eVKmrbZh8uK"
            b"hdD3pTeJ27+3mp32LVbd0GPf4pPgtOt9izhFGIwQotwr5ImXGhEeB8QFMQ4z4aWkjnCpuQAtlbNBEM6o4dx7RH7lyr8qV/5OFfJD"
            b"Q4LNGxKX+R6juWXYCU/9uQVMpzrL8SFxI4WlNkRMlwd923FT4T67gu5ECZBy4N4zfJ90s+WEAxhu0u1I97BB4p6qM9zjBz90U6yE"
            b"beuDTSSH1yo5/Mhgm28nDnYWV7PvX1Xyt7YpNMHeeRGwJdolh0biSeoq6/hDIo2l1kgOwpJY1mvmqIyhzFodSzOhvf13qvyrMjg9"
            b"tAsmhrRJ3o/UlAW2NnXxQx2HTywJbm0gDrZIXzSeBzuOtTPFyY5jvsUbVf+2EZizhCC1ak79lmO699U1D85B/WbuMBTIMo74LyeD"
            b"+I45ki5aQp/W+U9UIaDalXyt7jjnJc9dfHdViO7VJRZztQ74ae68/5m6XjOFXLDauYADAx+ES/BnFpz10gnLpcFaKeYMj7MHw4Zq"
            b"B846i6018K+RUVQv3/g+mslToDVZvpmdhvhRxuEEJXcNslyTnutG+C5E+cq+c4dkt3sDOQFz1QplHmbA4nWUBSEE2S19V5iHXOVZ"
            b"6ztLVy3xmGLA5bE22MbxeSU6y09QinH9caBdPaNT6UyB+cNrGSq3tX1vlvDF5rj2/4ELNupLpU/JWsNXRwUiTesWPCWn6pTjRg75"
            b"H+l++jdK/75I098mCjzKzjxmA41ZfEvkwUsKJBCibVwSc2ltLCu4sSgW9smTnQmOhRUKgaHGCQL+A4o/o3TNvlbLnxfywzS2ym8/"
            b"UJNAW/OBrNlCLA2QI1TxmN8qxYTTtD1TcrykVZGuk5HVZMXKsU7moSXJpDiMUMN2YSM+HilhcdJClaIG7h5k+X38B++fe4ujVvh/"
            b"8JTijJOcday05U139TEobhWgrN10Vvgdb1zX2iOytp73vjXfz12OsntBf7S1MOSPq1pv6TtcqkHfo+KwZDNDvabecQTI8OAD54oz"
            b"7pACUIAFR9I6obzRXGCjEg06zrnehPgapLQvstkkl93VLvtBiLdK+Ya1Qz7mw3GforR55RLC71gZT6pYupqcE4xKb6sgecXEBpgu"
            b"PvFyzWQrcqyvJElmNsOawhboMWo7FFnrXfDqqS7v2CmHp25BDJNr7hY1I6QfYLyRFLgnxe1hiwbqvWbFZ8v3lM/vJLzxkltOKz55"
            b"0uo9xRsv5mADjZu/LMmdED9epznU9VVXUsd5Z7Yu47auqxEsEEOURcwG6xBlwqb85hkSyIJUwnuBsMA2FnEx3znNQBgwiCBlmfsl"
            b"fly3eLxowbAulhmwXXqRr2M8pKKcDDFaxblrAUPaYWNzVPxSLXmLj2zFm3PKIPIZmHK8STxWu1nJWwk/3a3r7DCqLaw+l+ZsTzAm"
            b"R4+3nvLBt9Yw39B7pBoKPzYocmQaMyiJYOwW35+SAvOAGFaYt/5l/UMpD68EPqV63JPamTmtv9Bz3CrM9jenqr17S7Trre4/m/ZT"
            b"q1e/9I5vp3cYRZVW8cunDaZxqY68xwhjpFRQPtawkknKFHcilhFSaypDoE5wYxDWlolvbad+UZF3lvbvS/UMis+xgu4b9r7ERLVn"
            b"tvc1lrp9sfdVvrK3977E2l2gr6hzx7N6QMmlju8ntsaOPEfRbtN1bL1yW4wSig8fU3dTJah8olvOvwluvmUqOdtGw9NttDNTyVlN"
            b"/9JUsvOXhMGrlhzy4ynaRST4Rd6/0Z/9FG76et7XVGWDNpnqeqDOxx85rHzwnikvCEqSQoFzoTBQbmLZH2ws9jGKs4E33wuP2/KG"
            b"nCpa4DVBDrfX2GT5gF+4BLORH+4BHle/xxfgcfVdbPW6zLgNvthb7uTj43sOJqWS1Z/C3i5bGJ+h1/DM+LeXkatZ3Hid3tVrFN72"
            b"V84H4u6LqBxjvDqleavBc8zRkJI/x7gG8O3Md7Z+FddTUqSoN9rYxrL5JFRuBpqYQ6NnZvDshBn+h2Fy8v8Oy9r/2wRr8jeF1EbC"
            b"uDcSZrWRcLXNVgp7fKzZNzDcSyNhXCpZ2WfzLDX2UaRcLMCdYFJyIbWiVmIfM7kHxwITTluuCaYSI62sxd44xRGhQgfscczlVHyp"
            b"Z/MiHW8f0jgdw2k6fpGLL5jh9gS0y2nsKLhOFkf0MxABmfimXZgC5Lq1h3eliGKrRka7iTksW/tSfFCWOBpeTpwzVuZJH37ULuk+"
            b"5rGUW+WyPAhe3jz9S0f2eaOBlOB8j8kWjjWScXgoy5/D0pC+lH65wCj+QcE9ivUJDXrZ0qv0NfYjvJFUeMNS6X4L9S+H9OrKQ65i"
            b"Wm/Ek8PBy4r7h2TkqrBeGmrvyccfAj4Yg1385iKU7DkxCERsrJsd4oYmRnUg2ltLBGE+xC+r8ihYFTzGgbsggf7bsLTr8hVPei9D"
            b"B/Sr4pg5IgeyPXBZqRxD2WIXBHRXplQVn/oBMu4iJu0mgG5RFJ0ZNHdjyDeEMwEb4yGQbYDoQ3hL75wi2ow/t8y720wxEn0Rinaz"
            b"G0KqnscRdnFsZ+OpSCde6+uK6U3qbsih87GSAyedj7/Yev4H48+MCZ4YR0AT55lCMccSLb0DDopxJzAzTCvuuXSY2qRrFA+F8IxR"
            b"HLz6T7P+XhesF4iBX1E0Gm6tPtIrYrf0is6pdiO9IrzWrtf1imp+Ob7g1olPWHxJ1qNYdlJS3JteqByVuCwcsoCHS4E8BvMNcMN3"
            b"fZHuI+bO1YrkqVqRPFUrkqdluqzL6QYsV6kVyV2tqNGnu6BW9MsDfNDdsAwzCoF6I7SylARn4zeKIwwaNCeeaR3TFeM+SMRRkFSi"
            b"pMEcv5rMWCJ/kFzRS+37ufC9epdcEb8uV8RPZUSOHhgTuSJ1Ra6oXE3dkCsiD0kqZ3JF91cgR7mie1rz/13LpD+lKv+N2vFbAgMg"
            b"CjBzlHmOkUEmYB1wIAzhmLQcSCk4aE2p4sxoi6zRBCgI74hH3+d8/BOYzbf4Z1MG9Eqx4AdyQOPGOUAlX2E43Ja5fGGVNK7sxp6c"
            b"2yLiSPcodSYccX+Hoq7igMtX9O90xWXrbNVHIge8yLH2B7LEVxac1Q2ISw6ct3UxE8Tmjo/S2IETTx048RSLjKf7bHhDJk8QEnWr"
            b"dlU+xu/gpP05esYXpS8/Zr9ppUAcY8mEDVYRpoN0VChJsRVgSRyNZwb5gKR0SAlgTCFtQ7CexjL0ZwkBnfUDTpoBrzoBp+vtfsE6"
            b"agLclAaSyx4SXwPUZGcnvk2KkniRqlkKdKiyVyO2tjBINoOPPV6s2xAvSCQTpaDe4aPevzoAD0Ru6VJc8Nhdu2DgW5wja4+PBVmN"
            b"27p9cflQD5DDlLJYkFByJ1F+Tf8Xt0v5Wnf4fONr2awq20Xr2nw52pbbe+BfsfYuK+wRz+OTK+zjqlpaibAgQOKCGVRMgpjgwBiP"
            b"S2YIQcchEMewUoJ44UX8xjjHkQtGccG4/CYj4ndxceVtaYTHamRi9IYXWSYTm+OOM8tXFtueVOpl+wB+UBbdhUq3iwMNR5o5cmRs"
            b"akyWVVd7wkWBBLHb4QGTldYBVNhhcrh8unC+UaqQagC8tBUu7gESKTxlqjjLH4QVtjc7nETYMr62Br2tkkCl5EjeE0qYM37ZKeM3"
            b"C1GSCk/QHebfy133sj1qoFud/A3sTk2dxmV5/x9Qnr7Ysmr2qTb/uK1spXXZuivbxE87IwxHJepnULlrVlZEgFCOh+QS75WnCoJN"
            b"Agk6UGqtMEAUJ0oYyzXF8T+EeanBe689+3uEKs+hX+gFEheuInGhRXNeRn911LyFmIdPiXY7y45cpLT1NLUsD1yRzTZgahOZKYGZ"
            b"uEeKWnIZXUXy6+SaU9iy23Pg660bRP1JE4adWB98HZsGvWZ2iUl9A3wF5TQ3kK6ZPktCAFUtnx0C0N+AzDC4dMsMMdJR6/rbTbdZ"
            b"4loAWAYSf0pHfoYhkFNGxdlvZlheufUP5EamG/9sRRnIJWV/uCxmNcIL9QgvuIK5bfL0Zcwtycu69P7HsvmzmFuHDXdOO8qFYcoJ"
            b"CJgxY0FhzhzhymLsZRBpC8o46ZJ6hDKJYaGSWpn6F3eletO5eqjnxK67pfmZg3BrcCfkiMzFLrhD399ZynDcIi9Bl6C1D12mQtWx"
            b"x3IcIMx6vFfbIy9/i4BK/l4+lDaE4mIN+nAPKj7OezLsUFkUQ2VRDKWY3LqjstmGKtv0lWjXVqCeibR3dILjy23P/nc76sp2lAOn"
            b"FOXIaOyCDkZr570BZ3Ay9zTMSqMJk9xgqWmQhAQjLJIsMCKd/dfQqeKAlbxmR3wtJTzzapvYAaXV7EJ6khVF4MgIvpBDT7awsqlR"
            b"n9svrfHzWj3hV+keJ3cLtZG67dz+bTSMCTwKrYv6Bo0QR7EBoxRrH1gPVMsR+bm2kNQ7Vhb3dM2rchEf1GpHSg24isPNOfOr7doL"
            b"rbrCc+2Fn2tD9GLL6XsMigYoVMelY2lvySJtBQVmSCLPskAJZ4YThgUA2FhLggVJKHLUcohlIzAHhv9ljVbRix5cthW6rkJ4XW28"
            b"q8deSBGmMeZQjAF27S22b+q3K80clRJpXG5Cq17Y9QBKRPpX7lyBtWBtg6/Uv5fckiZy8KL3CCqjOrKrjuV0iclsMI5ZPTp60Awv"
            b"Iekmnm7mJ88jeU/lhjdtz+5w9YXgjTEE7yivIwLtsrDelu0VTqran+qP9l5p3zV9V6b9eZrhf0gZfM20XmiKFNEEmaCRZVx6qSjm"
            b"gsmgTFyQI2kCllQBVp5ibCjlTnAO2BuD/yb5mouubs8kwMdr28vqNacltHytPzARZiGbyo0qXdI9fHuPl6ixiTTPzqda5QRmFLGR"
            b"tdy2e3XUZ0xROTQ+ObI7vtEdtrCK5hxxBvk+83lH47eh2E4Zfk5BvePbUDiohG0Obgiv5IEV34sPEN8SRp8r2JD3yJY9KYZfXG36"
            b"alZEj8vpX/Wa71avcSHQWDvz4Ky1TghgSkAIJFbRiHlNLGhjeJwckAouZgPkkeHIU8EpYYj8IhnuYRQuAh4uIRku9hWue5h2/ZEz"
            b"ECytOiCE7BABNQG0LmYNJP3dLuLipGs/DBhwOXRpLGChrpjKl8BcOXeW8sMF1thM/i5g4ZaZ/CyTnv1mlpfPfrNm2buvfkEJb62r"
            b"Y75kngIijkIslWP9bGIWDUZhR40PFiSL30IkgHCrREDgPXJGOaIFA+s/IQlcq32dwGDp+5Cw1wgIr+Gul7muYxnhSy3hsUHaGdP1"
            b"Al+hFspCvZJxfv9WjGs4jI1KEP+FBhA7vnAOk1M2bBuqyoMfu1ke1MBWH0tQTdk7esYpJk0Xzwiw8JbM+jh/bnSx/nXPSDjPrCXu"
            b"x8Nrp2SErxBbP9QZ9piBCjrXoVwmq7KAhOBcS8s05kI4ja13QALlVFnKeTAi1q3MY+6E/DvICOcwrxcYrw6UdZeM0MG/5soAvQND"
            b"xVJduaFLMLtUgffyjmJxxBlSEa5omY1haudUhC5YFfOL1DfGUJELlr820S8GcEF4HpkI7AUT4S7SKt3tXRN4XDGwcOP9tXtDyN4E"
            b"vhbQan4DFfAARriDTtywcoLnfwO2qsl9FcDu4wiqA2rKg/BexSyGJSUuJjRvtZOCKRoo4oFIbQkF7BwViAbvJZfKSyS1F1oq9INQ"
            b"Uy9RByft3Hdx+cV1Lr+4iTWYcPnFFS7/Ilh4g8uPLkjH3OXyo9ut8COX/15v85fL/917/t+4s78msJizsOXaae8Mxi54Y51liUZq"
            b"qQTEAhGOYWuQJUpoRUks8AjCnmgPHP+DFgkHqPgOujqYsVz4Izyiw3c7wIYvd39jKl+xxDBMFRkbH7QDvuY9e3Td2afzznWHVF1H"
            b"VIFE5WjpCctA078tPmo24IUZkK6NKeWoQb5ObX1OAbWd60EMKsNhBAvYq3GxOxI1117i0r+sH00XW2LKuJ9vMIG8CcXitdR1fbTY"
            b"4/CK1L8dLHACqNEE0O1CHZUDT9Wzp3tMu3HO0cn2d7/pu/ebvPUIgtEBG83i/2FplTbGmoDj1ICk9Rhb57WIk4ACRVkibjnC4tzg"
            b"AzFP5Fy25cRoapiv2UfC3G35+mY3BFiQSRihxmhLLJ/gXvSWdEkqR7MJfGnNREdB6X4yGRocZGJRmp0Qh9YgYpERrq6XQpLjI1J8"
            b"Q3OpHSzWgr9QblGptOIiG69qzWv4YCSQo5LuCt5ZWNVtTc+K0flpisVhhq3GA6s9eDPCruO7QO8KtHb/UOXOodkpVvkxygVmUJsA"
            b"HR0b8hVTaDvdTkeSx64w+6IjQpzaZPLFvKfvIit9lv31zJK3lNu8lnHhddycoos3wizeyLGdJPeC/uoVYV9Tab9bhbtunG4l1+d0"
            b"uD+m9hLiH6jyFHj8etBgBHCijeTGxbRtQFDkDApaKoa5NdhIhTAnhDiOFcXAnjRYS6H4oIaHH2Xee12Ya8iU7S0sU4tR1SrZ3azR"
            b"cmT3PvTLtugQyDBDEo9kblUhqS7irKvSd6nbUdcPWSW3+bIn35tIXoUeH/XB0gByJGVIbPtcrFYT6zrUKS5NV0zgbl+MD4IFXgaO"
            b"meTi2CtuH3OOYfi5DhfhBAPcoM/KhuQKl6x/Z21gOW0D1xbBO1V2S9DNy0rX9acJcd1G6n5BiOtj9NmYdjnSUmjrNePYauEAIeSE"
            b"NkSZECRwj5RGhgrnrPVJ7iDht5gXjJGLljVzv5qv1MTnBfHrBsqhMzkxW7lpIHPV8uvI060f1kaczf2A1jGRHMi6U6PKIRhq6KlD"
            b"NzmHUxvGvSWx175nsFxSlaPyAG5AUCIx24geldvOcEglNBspMI7QwXKno/aWoC/Vu+nrc98BbOzZlZMqrzgMHb/2Lpt3Jg5z3C/b"
            b"FA0vdCx+vP9XBc9a6t/vcZs5hWcFh7HXRAbFpA08EGO58jKuOCGomC2FtU46pQLXjATusdbcSWUlEMQQ/wQ864tSLtNCjz/ybr+C"
            b"ylr6yOlfOaiPOjyQLF/wrSmrmj4y7Yi2pTlNiaSdW3lZiMMFQOzBe2GvRpcmMt/cZ47CsGeo2F6wcJKjh+2eDInNTwELzFk9Waz6"
            b"L7QDXq3PgTJFdte0Wv0rv0n3VPLVy80SBYytcrJinRG2XngzsebQ8k6cUbEgFMTSeKkVzfo7K+G47kM/gjZwIQS5h63N4orQSC1C"
            b"18PgdT1cCy1CI8MIrf8YNP5j0PiPFeoadJ2JSpj25cu657Fe7S9WnRkBcatOB9SdDriBpfiG/nRI+jJYiqA8x9zpWDlL5qTxHou0"
            b"CLQKxTlBEMDGSqpiZa1cEIZJYSDOH/9lDbHLAmEXxMaueNVcmJnuKHGlP5tlHxJ3MLmBMNk9R8jbol3psmVhgEaT5NHpMQUW/C7i"
            b"smG7DS0tS9giC0aRPCB++4Y94CWwAf2OaRtDyO/3ZH8OtdBuffQiw3NoNi+7Q7xl/HXyqI86X8jGVLL5TdPDPjS0v+ol+as0NmqV"
            b"BKmDs4pqjSTSoGJmlyKuL+OLWJwgDC44xg3xKCZ7I2lAitmYB4AD4/avaJVcbV60sLSSU9kT97IWUZexuM2YD8K5TXdm+K5zWQZ6"
            b"rohRb0jSoyBF/G+lRjHbZMtBxacLb5bAjX16tz0ol8D472o4CYt+BVT1Pu31IjLQRnJWN/zl+ifWRKegcvXEUmNrxt3FKEYPMUem"
            b"J/6l3klZct1pQOOqTVy/JpXgA2mEIHANB2kO5OIFgSsvCFy3oPku3Hj2cq3E5Xsq6t/eSd87ISguLGMhERDDgUnMg+A2VslSAhDB"
            b"LKHaY6cJw8CtJVoqQZDz1GBnuRTqG7Ec8NSb56kF5D3LHbHjeTvbhkuOOjccIFsHn6NSxLGD0egI9TdZWroJw7zdgDg9ZWzLQypp"
            b"cXpmEL9DNtjB11eWAliCHHRGjuISeQgpeNU4ZzsGe7sBPhE5F6glbUzMLlLIslB4aP+Y+mp3cdEjUx42hXOwk7YIOwF0vFY+bzV8"
            b"3tjGfisV7r4zz3My3KewGgRRKb23sZpVCb4anI4p2RBrA+EeAeImMUCwIcaGwKl1nIj45dSSOYtBfytW45VYpZjTRvAr2sgXtYDG"
            b"DNaLOjhvRH1cxXQcdc/rRnqPI6Hpe19JL6z32kUqvHS9t4uKqug5npAiFxpJJdFQoaUHHL0XApq3PeB2wJ08aKsX8zWEoB4bHY0q"
            b"30AJfSpkiZ9sOh6Ez9lU3vzsNzkJb7Bp0sOmOzB0l7KrDcc6e/+9AOkvqVl+phkRq1+SyMpKpy0ZDTFNCw7BYwVEkpiTaUzfViOu"
            b"lMSJ68eUCyYA0cl7HX+XmuUX5SFwRZkb4uXu6jVeVLl5KdiYN9xikSiqS1VtCTEvPPFaSYpXQDl6qEnT7+sWxE7Iub9cWJsUdOtR"
            b"jFXZoUTEf/jxVjtdyvQ4cuBa0N61jbzlTT4TVSgNAVLJMpBaz3zj6tWv5ZLjdvEGaM5Z+gDjl2/g7X0HLm1UhH5cJ/Kw0MdxhU+t"
            b"DMEK5QUJ8cgrJK2Nk7IQ0lnvKGZxlW+4JgYMJ8ohTlkIFBET/m4V3olkV+8S9kYpsUvqXxfhGxfT5wNXtKuqYmOM21DgJr/7IvjI"
            b"a7UvuhqdDenOWYwsXT7+S2UHpBiesciRpUYpxVWle8apzoF5YE+RvbetIpYmZqWdC5Xe7qixek66g4pmBw3lbr6WH4s4QKXZC786"
            b"vG9sqmJHMOYEMWsBPA4O2UBxiH8VRAYJwsWVflrMg+UxGTMkGXMIa4u0csL4vwaAcMK7eNp0Hfnp9n/F78EVXO64TgAIQ373KwAC"
            b"vgFA6FsOLwEFPdv7FLEwIJLnCxc8ATBAA7mx7oQclRELE/wBfi/+4K52+p2ilq8+vbXBL64oFHinvTUHdKFMb5iE6iiF0toUvTqQ"
            b"Na6g2RZrfl4hEWSNN5DfhDd4g+HvnJfxFU2yD63vCQlWcUXBMOcQ8sgBIJPYzDp+J21c6XMQzPuYEZzRWmkSsGGWChfikt//2+ZA"
            b"b/b8eaefz2Xfoo8Z/8ThCdyu1dUxUKxakZKpinO3Seh2d/5Jg6A8hMUhKK70cKHpNXZ4WzZmi13Zs5bqHQXIMzXHueruWA39XDVy"
            b"rPsIg1fv0tb9tQZqkLqExOWU1UzorCFBiVIGIRRzMBZeGksEUVJZpgz4QAFRHb9mAFZZI4nn4R80rPjCTtUwFQ9oHWh1YOA9MGtE"
            b"69iEGtkhWpWkjLlCsi0WD/IKOSRl8cRkgAbbigeXnjQcxiYOe+PguudDaTdQNfCUGMsM5+AyeKJY8Xdb+sDHrTV6Tn/B9QZZG12C"
            b"cNbDVvwo/1Sqp/YMtVGtOdrfgTXeGs0p5QZK9FdUhm6U17AIUO5c5dYEuGp1VFzlsTClPCH61ZCxCjy294w3wsbRIrgzEC5F+K+8"
            b"0HfLC8VJwWEnsoKQRIwQ5mJNwxSmnhoiQlBB0xBkwIJ6MILHA5brdKqlNeabJDL76eB8Lvhyl3k4Ccw0Kh9OFUOJza5lPW0yb8lG"
            b"TBHErYbcyNeOluX0QVATb5qavW/Rpeo8XXCdUYDJ3epi+WTl6gPfXT7beaYbpQqphgm+NbOH95zDy1vFaZKLPgt0nQ4u/ldrTdyF"
            b"K1CI5ZIg99C+cx0fWRBifG9N10c5tnaV6w7z7ytbufboDP0Lu3RFl5Rf6wP9jan4DyXcNcmCRuCZUN4xkt6aggTPqdXWgRCIGhu/"
            b"VrFUoloEDdokBaGkRaEUFQx9Avc7A/2e8tvOyW3jrnGrMnzeOB7SrgdqYXLNUB2BIn3/k1AYrzqw480pmYNUed9NXGcYCjlMiX19"
            b"P97uEjlGFVOKlLlm1j7b4kTubD56JPRdQjjQTYkC8RKzsbKPHME0thz4Ff10fs9gQlaEsvo1rmgK7etZJYqn6C+815wNSfj4059D"
            b"F2vatBPRs3aD7QIp7D1w2TVpBU+CVtIzlBQmJYo/UYFYy6X0GDGnFTZe6BAs5kwGijGSxgeBsdPMfELo4aMMhKv4gKEC8MRTp5fa"
            b"6e7rutrOLVRAmr8X80qC+cF9ZwAWnYj00Is6PfQg1fNCUoceHkIZavqEQFGAoQfQsS+TQ3OS3CWWD2c1G2NpRIvKcS0RvI1ucEoq"
            b"kss2Y2kKi1o++birtwQe9JAHVx4qEN/beyNIifRR31U/G2+vrUINjWrDhc01WCV/KqAEPxah1cuy08b3OpRX3YO+O7D97HWf4Mdv"
            b"w43y+5EO3DQKPmkZdLlRQEFoi00AK5SkEvtgEVipDPFKcRo0IZr5JAOkhXdWIs2AKmWCNtxh8jsdXJ0OOrYDvcJ32HNwp1rw2Xnj"
            b"u5J2ictJmBFohsZPUG5kiY7/Fo6bGJ11sDVOT6qckjafSa+9RgcTuFhCOUNIrYLLvD+ll53KHyAjy4fIBxNkLxOUrp5OeDhZ3LMM"
            b"ObOon7sYjVvPbLoUqMkWuy59hZ4bq7t1L1uN+9/p4funB84E8ywo5CRlwsm4TGCUBuyJMhbHJQJ1ASujg1RAjHImnhLXDZThQJl/"
            b"LhtxCtU4IWTQV3o+J82Oubg8nsnLH9w1jyv+C+sAuVTgfA1Qm6Fla3MfI4REB1mbphBGuV8C7QYhr+59eU+AfPdLB2Rqcwk5AiO0"
            b"ecdvBXWnM39Uyd829DoRHcQXnXwk0P545da2GKjk59CvNDjSx3BP9H3kq7SIOmypsPVYwhubuH6Np75Kpx2OXfj958vhfKW78VaG"
            b"Bo1/SDqZvHlPGLPOKCoFCIN4rG9tEi8zknMKTBvluUDaB4qlUgkJoTV7b766V8ieV7EP+BOXEs/BzHLwhiOZm5EN+0y5hpwr18hJ"
            b"nhgq1+wlYR88Vq5BFfO2Td0l5S75ZweaHRHOIoeB5BVRWVTR7YVfSOKsi5L2oRR9m7THh9k6yy2gtuM75JGUwKdyCukbdD0ZznRt"
            b"8JR+gU90bfBU16bxudh2tmYvd4XJb1KK/O768A9VgWsmlWl5FrOk9QZL6wEZZbUSCnNICxaibSK6Se6ACRNQPCBGgyE2Lkec/qtg"
            b"ZV9Exr6WIfsK/OyGN91ljvAlGN3eNsA1bHbFvR1UXnJYbjYQggYw22GjgZS8lxyu+ABpO2ozsAXxnPkJ3Tk7XKvVP8gE5bz8h134"
            b"oaG5deekh1jCGSWY7nAHNTmhhNEcD6L+MI8tjBJSPvWn4LFHJLqZentBH6wN4/aITe0zWMEpQA1TaKl0MNB5H6z/d65GIwC/qwGv"
            b"ggy/QLLvBpJRBdRzLA0yViEtBCOKY0QJQypIbRWN00FgCgsmXfxGBkKsVxZZACMM/a9ZbFx3p7iyPfjHTDZuW2eUq2ZaCOWyMcPY"
            b"QQv85Tpm4r53tEHiJ5fNI8ix32WyET/ym26hU1/PsZDDM6PnMYO5SrDVy13w5lWm/dWIfFROO68NFYrGqhkZgWkwEJRCjtD43Ure"
            b"zqB5MIYwYrzFMVAYzWLBzUL82Uf8Nb6ob7MzeM8aF/fFIK+rOA6Zb+MNsJ6tVm+AiSl7Wu786Z7LsBDQkFQ1XWLw5s+6MLu4hnpB"
            b"ZMMr1RldJrLtXOQmOIUU6XTC0WlPON11iWpYyHIiHPdFHnJZAN1pV4yd23BB0VY7XLzDp+GqyYEbUfPNz6g9qBsW1ctN5xHvG1pN"
            b"r3eJ/fENiw9oP36MBxffK1iuJefSKy89t4FbwzBY65ADzQINXBhLTfzqKu88ckJhJ2OpqpnHL9C4QyzumxR4/5BPxZboZuYTSSgx"
            b"YRaKO1BZL8tlBu9UGOoGMV0bxPyqhtAl86ahhUYWySm1IVpBv7RptLa8ZVkCa12dSsuhKeRFydtUSNmGNreeHk4OASQq3kf5JNtG"
            b"sigZB6mtBbHvYrVddPjSTleaO+9p5c628PHStd0TIWxlp2w1GqqDzdQealP76nf7Yn8st/P3oHrbPFgZmFzE9aJOVQEtafFcl6FO"
            b"iJvmAgB4TgKKORC7WO4QaWywTHFCJRcumT9oTpwTlAnCpRFcY2x4KmOEpfZ3of67UP/3F+o3Gbezhfq54uJ4oS6nC/Uz44Yzxu3v"
            b"Qv0TC3UQgiMVsAFnGfeEgvEIIRvimt0qJDjWgWDiKDDM4l+GS6bDCgewmGsDf9jM4asl4SXm1kVPh0t0qM0Hge6b68eLXa4b84Vy"
            b"Y0BugM2pymKKyaoCkok9eHe1PGa4tIhNy+ne6eHE3vKyG9zlIprLqp2xtgG2HgA/yHkX5m5uBZANlrqBuLah97eaHko54Uv+ZTdT"
            b"7jxJzsrU1/IHY0cecqAhVLm3Uzuo2bR/Kdns695kH3NtgDi7h4Cx4rEQjWtyhp0nViLPAkaYCsQDp15A8DRmW+wcA+NkMIpR5Kz7"
            b"hRr8Qg1+oQYfhBrcFoKcQw3kKdRATqEG8gRqUAsetCIIL2Qhq6p6hxrIX6jBH4IagLAIkJEqJMVfZznTIiCGqPHKK+md5E57Y520"
            b"1sYiXRPOHEECgVGahSeVNzuZHU6nhnfPC9dEIatsepJLr/vvjGvv7k0P89FsMprU3mImyTjQ7R1m+3y9pIgQi9ipF89B5Le0OLbZ"
            b"5GRq6Ofk4Qw620sbzhyDvbRq0Aex93iFbTftkXzNgxr7rEVBquqbNDX20W7nTJasqqfrFNw1MMaV949JvyXNvvBFe5NG5OcqbCnA"
            b"BU94fFMekrUv92AYY16B4QZJ7RR3ImgjnZWBCwpMxMpbaZrUev/LvmivC/YL1mkHtfX6Jj5krwZLw7mid431H7DYcBQVmGFPr6w3"
            b"jGzT8cYx663OHq5p7nui5aRccBEg111A3NlpHsaUnkw54QB4GJ/whkR9F8rQCKE3B3haKuOy/VbJsFcHS9lcKT1W3u4duWKMaxhA"
            b"HHjxifo1SXuriDoEnjzQQFCtXNJOl9g7ZQkz1EjhwOjAjEBcIsq5s5ZJL5x2XihJMFF/nTLDH1HguSgHcV965yJH7wdI9NwTk8iy"
            b"ugvgAh2E22ZaEjn4J2o8pMe0ajw80l4ArCC+6x28xZnPcG5s8KbNwVtKBW80ePYjth1VHZPG86jjVRwhGM3PZ/G/sgw/QJaBIaxj"
            b"ZqOYExJc0BZ0HEmQngbsPOPCep0K9zhBxGnBCOEQFo5rKhgwg/9uI7rLjm9bi2Gta2fbixdMM4cul+R92sTj2QEteX+5Ervu01kb"
            b"xHdOnIuy5Joed1fPo7tdjsqwjopTPbHuzI85xyWgRuXcNHLmLCEiQ+I2u+Tl69XeWPotPLaaexvgQ1bs5vb1vJsyk4qYdUbmgI9K"
            b"RY3/2sy9E+7BwFNQILgDpClP7p7MhFh7O2eREyxB5hQYEOBiTBI+Q4xppYVy1CL4T9vMXQDGXbX6vMz5aLcdl03Ho6xmRwzZMR3H"
            b"ZkYRLcQEDTQjvtFpLo8gaeJgCVum3XBzh7wsi3pPk5fHXf48gDUvr9GVuERvB5qDUr7FK69jbePt67TSmXjE5rinMzyzJPrAbx68"
            b"+rWNe2vHg/GgLAtSJMqGC8FoL4wMnjOKleBYMERxYIrhODSgICQLRGkK3KMg5K8W5a8W5a8W5R/Wokxfmjv19liL8sy4aL7fOEcI"
            b"7mi/VYtyChA5VNtj86Pfpsf3Nz0ENSj+sXmnEQ4OC5pQg4FjC1IKLUVgQSAlfdDEIamNC0IzrxCkRrn9q/Eh1zAK7wORXAZ9vBtt"
            b"UiOs2WJexDvzoiMp+jqU5DpE5S6MZAInr6gVxw9shMxmLdnnuGQZALNvawfd4wfOBCbmzMEZJPtMZLhiAFZN5vFPV1D2jxMG+ifg"
            b"IjFdkuS/ZX3qMWPrpYmJ1PKYYA0XFiAkRznMjBeOcUdASYS5MIbEeh3E391ennBNHopwZoJy+ojlQhiHNXkPm8cDA7mB1OaFFvPl"
            b"Jvn7rZmXdEwRU02zYeiffOzVnCbMfFFcgHVEVDzqHVtHBpZ4JJMdZO42Y7Y/XTEUMs0xXD5uOt+UAzrLiGNnJPa/VXqzluHcM+9I"
            b"XHNGtp6LBEH16rfl/MaWswPrY21qeKBgXFLZ5LFS1RyLIAEU8rGmlTgWrlRaizX3CIii2MXKloD7vjr2cUMEv2iIsOsNkXExuW/Y"
            b"5QXtoMNxiaO4oS9a7MVLQiQ7+NPNEGszdMgyx8jtFng9z/T4ttrRc4ttppD2DLbV2kuKZMsWHFkdSbsUnCeBte98loCF3DSIaJ4i"
            b"1x3EDZ+3Z+DuLlKtS4uPKUX8MC5xNMFLwy/BTzPxfSkMyN+tvdSVlcqarIpg2XhjjNUx4X+1s1KdpHdHjN0lYyzRVhliLJC7H5OK"
            b"S8p9Ueo2qkHzUvdVMv5YqcsFogFbIZCKOdZRFRiPpWxQKDjHnaFIYQgCEYK0Jya+8sntQoBAhmH1Nxtc7JBk2iB+Xyu/t2vYFCDZ"
            b"eiW2SpKVXCDv2mVMJOnlpEfA6z73iIpxbSIQm3IFLEG8ap+rI6CZlVBKiVixGXxXhO9VfS5sleYr5Rb/V0wu3ksEGdGwZy4Xcupy"
            b"0bRPxy+/Rw/tH/K54DJmIcOokSyRjSRohjQKCoRLRu7MEWUVCyFOuEYKpZV3UirKLJHJ0OdHwRamqe48z72Qtmg3QnoCxsGyeE9N"
            b"JTG91cSng5IVIx+JF87BkjWgKgKb6y7VoqKyjxf7vHtfOg22IpEP4AzioKvBCzphs6yvhSe7IaSh5shVFA0vYmedJlo8O0d8KeVx"
            b"CZTBDXcfWaqQvXPZHBXLkR2T2x6VSnwnTbRHZduq8CG2zFcfFfZxzgerV+Z6VEC2637Ta0rFD8mIX1NA+wqU4AAf4A6E4twQZ5BE"
            b"2nqNrI+VnIn/GE0AeYGxpcJSojmnXDvtgwsUxVpOzu3Y/0vwgStSZ7dRtS/ACA+4FS0qdr36MNmmqJhv5caCO+dC5LtK0ThmPLwu"
            b"UkW/Z37QLP7RKIWCTyj/0hX1293NUVquBBeEA68hadOnUJ5ZCX8IH7jpTERqAhtp2Wxr8m0ycbN+hwpDsBvYt4LBdZOVDChx45ed"
            b"6m8tALwT6X4xBN+PIRAIQcz3zBppHGWcaeSEj8t+LoMgnksZC2aUiBTKeUECp7F0Biy4Ch4x+U07W19UZi/t12nj9VTx7RJbgV6W"
            b"aBsLDp+Ihbf7/3whY6z32xWwQ4Iz7Xek8ia+pLwK29TVxHkTIC/qK62f+aJi2kLJ1VHqFBCmqhGspXNbZaeYNNK1IG4q98/t7t81"
            b"loBl1l9X+qRqeY57AM3m/fjlkl//eHtzzX1VwVt2ml7vJ8UPqgDqBo3Mt7cARFw5keTAE7TEDrvgHSGxzIvrRcywAWmwMwprTgmV"
            b"lhruraGMAuEagqA/XO/8E3yFVu8cj+qkjIsVpGMNtH+kkCO2PQy1rmyhTw8pJCaUFdYzC5wIP9Y3+BKZcJA5T1EEwcY8oG0dyo9P"
            b"BxXEr8B8eBI/KFumyOKSLAAN1NTzWrBNhYCWieQoqt51G1ZRdfWkiCT3mqCkogKQBk86a4+SKo50ajgznZwdgzp69dMIBdPNnaXC"
            b"61QnL1V4b9U7F9gzLjW1KCa1kAo6Z7AxjjEvAVGBHQGgFjwwE6s7bbxyYDiTWinv8LeKRlYJb6pdQ15p17xCjN6XE3uK3cRz9OaB"
            b"JUuGRkADumpjJlShNfPbVibsSxu6f+MUk5I3pRQ3qCN5TOApJFs7SKADhFKvf4b23f1VMX3f8ZYj4518YUj/Eeu6uVOk6YefIjNy"
            b"lNUsW7Ip1/fhMa4MJzHA6UI8YOvi//C4Uwz+isDYLfN0WcGOZOfvC02u7PPukXm7mqfLqqR8bb6zr7fvW+/8DK2aD8BFP+a8I2JV"
            b"KZzF2kBgTIDD0hOsBOVKYca40sHH1Ta2VmmvsJFMGk2dipmHWWzfu4feK6Cfy5/fb6d+hfV6la31mF41WhIPLCZyxGsriplHBhvI"
            b"V5b0ctyRn53B1+2nBUWESdW8lEcKK5RIzABQ41qx/9F1Z2UhnBwe8x+mu0nZMrLZeSU6w5W+onueGqD8piskr5uZ9dEK++K1d1nl"
            b"35t1XiqLsv1gMeORtRqYbCD30Ln1VuCj2qN36V2y7+ldvmFjqpa+ndlOLCt4+i2b+KegUMEsIMupIU4x6hwGqhhmJrmasbSFhbwV"
            b"Ila8gTpqjdAcBOcSWe0A07+P+/pZ8YAv7oPNUPHdAMdOka/IVJ2/4yH73KTTzrMoHwDh+b7XJhu/tAVmdTwnawLkseC1/oc1kRYh"
            b"mDHHNQfnbi7uWiDNaeQoBJ9GFv9tDSvJfIQlNj+CVqf95H3KqEr4w62qnOBv9Blme05yuuckl0oaqkr69W9Kfh44W4w6DkPhGQ71"
            b"y99dqu/fpeIaKSYVRlpgqiyJVbR2QVntETZWccd4rK4xotw6GQh21FgWuBE+fiv1v4pkeOOc8eYMf4/FRCu+ANngpiU3DrfiS1wh"
            b"MDHAWwHM657yAaqfxVwgl+ZYqEEKngrd5Jky/SuPyDDaz6mlSKSKj1RuaD9b5emXEklJ+40R+x/htvqQywKEErXo826cClj/0LvR"
            b"KL5Ety4fJ7dbHk4JfzoTPIAsHDhdeGlEQ9UQgWbv7dhUPmubVHCFqvuxpv9X+jZ7E+VbTI9+J4HRJGB1zPBaOqkECQDxb0clPILx"
            b"sfxnUnhkicfKaS2lBOKDBKONwWAID+TfbnBfbl1f7EjLpT26snQ3ifJh9HPQRG+4dLmxfrMVXnQIrrS3L/biRyoLJw5N+b1xZjNg"
            b"XNN06drPHrjbo/zFS9ePmRzRZkRwGFGJkeibVBBI1cgmDa92ztKdZecVJ0GqpjY5qiAM+bhrU/vd2InfpnaXfq0gVFNhBCEGC0os"
            b"FcwRAhBLb6IUIZozgqXWFAfOmVVMmZiUvVbeYvcBpFit9f3WxHuSdV/AneTBxmIgYLC/xZQxVuLkAZJ1RIxdkKqBw9AHoLISt2T3"
            b"ltiVwgjbhlf1s8ethGuqO2wl5W6MsZVhdrCvfj27HMhiN1Mfv6fCOIZB4ClCDE8RYniKEGvK1PHLvzjLfWMuW/OXi0Whk9RQbxmW"
            b"iiaSjPBUQ5w2MUIGGDVCKB+cCNYKYZQwEAhljDOj/8uWPy/9fG4AJxqhxYHM4oahQuu+1tY5oPhEjrECEzTXY3mHhxEYrsZRJymY"
            b"sGwK7zaV6712kQqX8eEFXAGd3c8B6JURFumuFkBDA2aoT8RteyCnUYIx6ZYFy3KuGlKKSR9jnJDRvhisfOm7+BS2VLDASo+4HxTt"
            b"biKDLHLwd1WYszpytv4/+81MWrGWHFhbxeOf7ZXm1zVgfu19eraa8FZiDWCExfF/JK7xEZEIU0usoIEjyqnzmIdYeapYORiEtKcs"
            b"rfg9wfxvMj2+6D78zDhhrIR43cj4okPxLVfma6JaN02Py0bcKq3VzChysrb/uT7J5RZyXY7qCWi/j4GLXg7d9uyw6lMKNEK2KQDz"
            b"x6bH/Lanz0iU9kwjUTZW9aT7zQhAd1CGqZf/3asxyWLb3mPvMJP49Tl+0tiVNDE1mGVSeiQxU4pLwNb5mNvBmfh/llDHHUaBAnLC"
            b"cR9AeMG5snGC+EGdhbIYpict3eks0VXJ/6P/q76F9V0sC97cw9xCeCU/iFvFrkUQrAvd+cHsSu+zcapJAapqB7RgveMZx5ZzeU71"
            b"/SiWfz9tMxzBgl9vM/DbTexjm6Fc47rd5W2zB1I1A0i1MUaqLS/StAxIxd0l3fbZ9tf749oM2xYWga2qWhKj2FGJz9oM+UMaJjn0"
            b"9jaDBA6BxiLVC4Zj0qJeKKPBcc6Eok4jHQxjNua3WMJqhpEByS0D4jlQ5X9FuT8ko93uFzVsiIfa2VNHGzaS2V76DWrpF4jGs50c"
            b"/GmKzDbfuxZsZvabo3J+fbcPexrsZu/LcCFV7CCODmkQAyh53Bm43YAdt1lnxSWeCnDjqdzsKWTg8NOeZfErwP1OVUKF4pJKEmoQ"
            b"dkoIZCgOsVjUsQbkMuZRjnnADHsLPv5CO6eJgyRtoBSj5AuMioVudLKNNNfkwmd8C3a1K3DYDTmQJOoxnhMV4BHseGLJe8mP91hg"
            b"jtG8Q5jXC+teONx3ueqSO2VDi8Bq3d3iL4XL96ZpM/RNDaFgtcTaNBhfNo8gxyZvR9Rb2vDOdLcEtQyLXPt+TGZ7vhk/qxWfiSPM"
            b"N/BP1V1fLscvAK3k/x3Ah/9Xi4aW441XgXteBbvCq6gxVDNexcLp3uFUZfkp+1yayT4f5VUoIMoGZqQRMWlaRa1lFivrHObcc6kB"
            b"E+PAEcWUJoLGP1chBfVSGhI4PKlPt4/r60oJn6VgLJm1Ws8+U9HuZBNq6Gu7jyRKsxXtalLNJlJTjK3MBip4nUuGQgutivfIPj1f"
            b"hm6ehpVrFgwFCFIkTraJCu/p6ajNhZ54WPJlL0/wlqkoDjqKKSR9hgLBAf460GNAezmvpnJi7W2mkjcHP3RzlPekZko1Wa/qd3EY"
            b"0gFhd6ODkToDm8Jlx+3QsY3M9S2uH6O/MBXXfm7o+LkyFqg3wSLmQzKI0Ro0ti4mYmWEh3gAyulYwcbKVongsGEIK6pRgJiQNfqr"
            b"mwNX+3UXWgjvbg5ccgC7Y5V13S2sQYdtbgRHqRsuK0uFC+Zbi4tWAnm9vS0g1d4WAFX+oOT/1Lw1EIMofGdrgFRl6BVs1oxNMG8a"
            b"7LyB7tWwkcp36gD/bQ28OadqpEUgkmEKNNjUfg3CSa5FzKDgGdFKcmZ54Jxiw7XSQRGvwDLsERef4Ib1kgvilerCy8Lpw8ILZ1Rg"
            b"MdEs3Blg5KJtwcLxPd7wRfLZCB9wItF9jg/Y4VGv8AG1rjc/lKQzfEDNsOI9pGCCDzjh55bAxVgXKnzAdh8Dlly+7XICZ7GU2CUf"
            b"ZpNcCaOdXO1dqYdborXPDGxnWAE5xQo0rN6K6/uCCtyxglsr3H9M9KEqm6HWL4c/KAcxwAvEGpkjLCWxjCvHnWWCWWYIokJYRq0x"
            b"QXgay2qDBcHW86QeQahVwLQm7ksZn23C0l9oCYtXXeHXkrQHXu64KSy6/uiklUsvdnPpsaF7sXt6sbl9YyK610peJHPS21PaNFDw"
            b"hvJlL4QktsDB4obISkgCraPmqyXF8QxapowUPFYuHw6rjL6ccNEpPX2Mq1M6I2ydA8Qm/HPQn89h8TvUzgF3m9FJUILchYxtvhLN"
            b"QYH18to6jNfeYaXlv2V5XlPR8gFfu9z1UQP+6l4u7ZDiQXGEilXeFOP4f6xr/XJi+AP97OHEYJ1kTDiCjJKSB2OI4p7FAkhzxTHn"
            b"iCmihEYGa5tsUgSTMbtqwqh6hr141dueNba/0NU+b2mPCWotPW1i9XOu6JDRqQUFQJfmR8GbbRcdnHSpX87WNg0mtFVlmEji1D0T"
            b"GDnW9ouPwcM5SgKP5RAuGV6mq6QAJfZOy3Ds+RZT1DMh35tC5ued4tFvyIo62/YEX/9mV0Kbv/pL+8jf2C1eU5hnCdQgnEsrPmSB"
            b"GAEkxOwFwVqKdExp2sWBGMY9Ry5WuTZYEU+jyInwiW7GF80XSMuen+QtdTtrXKqKR2o19KCwlWpTwAgNxF0acFgKkek/ckgmI6cC"
            b"OGu52wP/S8jaVcZLah1pWab3zYWxFJV3g9q7ck35movtNLu1FnGqDcwB6bOQVB721Nq7T++ao7Ym8TqAvooft4inFhfHDvEfkpuZ"
            b"4W2hOoJB1BhvOxaW6ZV4V8XJv0ZY5gP7bR/TLNDS08A5sZxRK4IQRLpE8+VOMK5iDnVEUx0sj4lWW5uKQYMZY5p4Khn8s7ph14zI"
            b"LjWY0w0B6VRue0dueCg7OecCb/DuGVe2QxHgZ5ZtE/3gYx/kyL89YhMO/Nty+1e9zvJTKMG3fMsWDzeRPdz45uHGa8jHqBe+vk38"
            b"Vyz9Fd54xQ1PK+9RznkoHybvV7lj1MMMxTYnFs+vtle053ThVo4G/oRW8K942GAm0F5roazG8Q/PIuFImhwCYlaBc9SkjoCxPKEw"
            b"dcDAEWESUY5ASMel+gDygo67yw9mC3zWGWDjzkCLEOsoY+v8wFbyWO9PmVfXiuzfm4mUIMpRBJEKZYbnBpAshxIkmq4mnZ9AyZL1"
            b"Uem5Nimwy568RBGk+Ia229foXXZOQXnqiQv2Fro7XNanIefIjWVGV/dJ8qyjcGCa5St+jGk2E1rEjVtF+3rGJ5t1Ck6ZZpdcIL/J"
            b"8Ox5q6B8SAOzs0+0CrTznklONadGUw+eOxQ0ly6Wu9JJ0I6EuAZLWU4bpTxx0hKqnaEYY6f+YuDDtfL1vDYdghb+DKTitsT5mT3w"
            b"EKFwwYv3iFEA2DAKl4118w1s21OkdabY33K4MLhca0/tOo6hy+YXSZtZ8ZZ6z4uz4ZQTgHwNCvFE8WasUTMrZueKN48BwR02+ERn"
            b"cdn7+sVB/AkchLEovjcPhiCGjAdqBXWEUuWEJV5oy2zuZwAQpB1D1mLP4p+11yj+K/46HMT7MQRf83e/yszrQBhnCRyPEviJv/sA"
            b"4pHHmYkPatOQYEvh3TA8OiozzW8wtXmHNvHDjQkGr60sdcfsPZ/VdKBfmr2PIRMvzN73nUt2y+z98AxnZu+3kRO/lu+/QInPzhwW"
            b"OaQc4xZ5Yo2WzjHqOfJOMCoRc0hSykWQXoEXYATS3rBAuJNCSaq+yfX9Q82Pi6uMo2rNNpDugnu7pIFViSc2HF9zO8JVXd63fBJ9"
            b"OcUlO4ilV3JmV7Tg5VIf5EBf7rdZL+jUHYCHR6EitnKg479L7E73pt2snGMyY4VTQRtQ22DBkGNwRa++2025ZaA8I1bLaRUvp5uP"
            b"cir2I/ds3Lzsoc37T6F69WOaLDOVs1sd6EvNl49Rqi2WlEvNQOv4ng4ryqQzwSkfsBMgtVcCKQ2IGkOMkXE9qpkIzjCikDffqiz8"
            b"CuPxzIEev9ivZDdM73pRyKtcv4FKcd+dqAXbFzxcD4a77LF3z/5hrBW0nbDRY/gLU1B8bmdXWsaiSMlXIpujO6VLG5+T9ZHtc9zI"
            b"ho9s04dc0606TCG9RKdk5QRAkna3DSOhpBxXHtLjXM2FEORe/5vn2X4tc9uj/63Ej4pFwtff0daWtDtM1yg/Wsvs+givnqXH0rn+"
            b"eVmGLAcVIBm/br789FR+Qid8ncw/JENsPeAQ1342BGodkY4rhq0zQmAX62FGnTEIeLDOI6EMY8DTXg1yEjut8N8kQ3xR6feqxvBF"
            b"VeMbUsR3NIZHqXJrCRxTJd2tP9k2r2wm9tuJj6asM3zJQW/u2PMeUnIKfCOmXCQb6N865PFDSdHl9hjFqgUNjmWI85DzKoSuRfgZ"
            b"0b2ElYsfKYZDPviRYniTD/4mEPRjqHMPGJzCQ86hIE/A0r8KxB/qh3gbV5SKCC2VCgDWCmMD9sCcI4hiLLHXRijifazvKSgDBBgQ"
            b"JZjRBJ7Ly42y+ofYIM/aHvx12+OwDfv9fQ9yr+9BVs+1m32PTrbtbdp5s74HrrhCr/seg6d/6Ht8EIj3TE5u3PVg067H2T7l693L"
            b"fwx494fgdUvSdME7ppXH2CjuOHdMcBaY5Yoi7hDHsWRmQRHJHCCLrZYYIYgrPAdCGvNNTeQvsldwZcE27Gnc7AMfBNaXOvNV47n3"
            b"nkta6Rn5RkpBKCoZCNGC2RLoLUWtWZ9srd8OkZzSb8a8UUzO+r1ZBw5nSXXSQDAWpYyW3ZKdmimhbYULr6HmR+11uiLicPNVEO0s"
            b"hwul5wlqLqtsyFvSFZA30srqfn9dRC1kJWohK4Ly2jdY+ST7kTzZsjvY0h9fbr2Cn94TeIyge6tWu8fESSDB+cARB8eABqo1EURr"
            b"o5CzNhZ82HNrkYgDSQKYGlsMEBzYq2S7v6Psu1TM3SrSThQXZtgxuvI94is0ELnEf064+IUUMW/N3tbsjBkgebyRQ/Wbw1Jjj4Ho"
            b"JR1G8O0clieJh9LCd1bb8J69ryVR1amsklyHimwH294XLDo/axId7n1tjdMmFVY/3eAH8rUC228deLsO9DSmw6Alp0lUFgcBVqUv"
            b"m3LeasY9BAvExzSLmJCgiBZGegFaO9Bc03/RGPj1ZvehVjzuWrMFQvYl5x5x27lHXDGLv+Hc828YBN9DADzZzZ9RiV+iBvjJy1+H"
            b"4DvloEp+5hoZQbE3TigXCHhPAOOARdCQPCgY8ohibS33jNIY6kVcNAgkxJ/SXjhH2qKXLIwOa3pFe2Eu7f1aLfKeVGRPlLgmqsCh"
            b"UsslB45Ef9nWlHiVnzy1G64YsE0cw0V1l7NdCWZ76+WOOuYyI8swMUW9Gk3nIoyyCrzMGVmRfbiysSrGnalxCX6uxHBb/vEmWopU"
            b"uy2kUn88+814h+b8N2OdRxi8ahPpj2c3XCPlFiQq/RYOw+lui4/lYlxwB2+FMVJp4xzjgQtMlExkHRJM4N57qRhyhktNCSJJ0DE4"
            b"KrR9gpfixXALncmgP0vMJ4Ap8iox32fIVfl00959wGE76Crsogqk3cjOLucI8L4VPAlNMSlxLAZreEmiZZTlFDgg+ieqDWK9dfKE"
            b"HpZCEggK9Ro5A23EGFTUf8aSOvSywkP54yK9wkOZI9FCPGi1ddrL5xGUSMxou8s/UOPJz7kEfilXM/xWS/ajVPqZFfCcujCzCsqr"
            b"+k57obbw5WUh9DcT0r6atD+Ei/LBKFDMx6qXU6O094gRFqyPC35JbLCEBiY8xlY7AWC9QBQoFtSC41fZxn+4V/oV+m/X0ruzRX2x"
            b"RTsok8+kHd7j1la2zjGriFOqRon2zYKl6C7M30Zgt5dMKOYUVB56zoObSkPNg6CqKNguQ4ZW/LwfUAoug8GcwED6dvT8cmQa18OG"
            b"aczhMvkUXYefyoIJ3RCm9VHGLPFd0rY9KvmyQEJX7GpzmF7IivXVHtWY04raNWZ57QaXBYT620V9axc1IC9iWRyY8lhKHosNrbDx"
            b"hAjtVWpKMEEUdhgpbD0JRHBOkcUK2Vg4aPaXUbLkI+GHG8JdJSRfT1ULg2UdcEFKssd0Xsz7J6yxYzPj0LFdUlmvZrsaT+5pGJ9k"
            b"vRSVUU/bNr6o7S4OPZoJBrUDoJYHuaqI0fXCqqIwDJTA8qe4U4rFGs1WPCk9MGjzsFvw/zrkjzFox3tWeLpnhadGmOv+Fd4gAWRk"
            b"J9zl1ePLI3v2j2/pr0m1qmhLD3dU0ZZkO5dYWEvd+Mni/OEOtvw/m28xURzxYC3SXHINPqYf6TVIFwSLUzgICcwKxLRwFDuCtbDG"
            b"AwQfMNL/qjTkDXXG96lIvpC0Gbibvd3H4sX+/rF1cA8McU2DpydUvJJIOLaIX0okjFXJyS6RAHIHyXZnDQEe5ZSrXhSy8qJ4Ig35"
            b"rF0xbDDMgVarFUXtRLHLQI61EZqGxZbdZy8rm+O6e/ErE/mnZSIDYUISQ7DnDpAwXCpqHPexBBeOUG0Bs1iAO6aTczKiWhJJFNOJ"
            b"52to+KvYX4+4t0NG10WC2A3m18wRrhsgqVi4aO8LiMV/gXT1K1rgaITssSVyKAt/MJs7IfjeoaodtzGrP9LBLDDwpEOn5p8zTzq0"
            b"lemHdylmeenJxCpc4cpgbgNyDLYLc2h5Dx7nAdHcDR7NaSUs3/lT/pekcX2M5I0CP3fboSbh7im8pPcNjFsdZFxuddwerTV8g9zl"
            b"NfN3VNaPpXImS4C1md0c/JLDvpscFhgPEpv45sZRSmVcL2AHhDKPreDxx9xZT21A1ErGhbHKU2MxFYHEb7P761YKY9eMS7zgsVTC"
            b"21cec8DIBV4wOZVQGPKCSaXniyYCCV+TknusOz/jBdeYk3u84Ak4e8oLnq4N5rzgG9ajv7TgH+ih9N9ZDuhYEnnsLPGKe8KRUMwi"
            b"G7SL30WJmPfeYqyNsEgrRjiL0QqJIIORUsA3QZv7hcD5KuB+fX8Bq3ylgz5rmIyFBa6KTlxqXs+68qPt05G2MB7oMfT9ohFT7lSU"
            b"ot/jXJRv0t8vkqKRaBs/oqmiWwk/ukFLsTCb1+76zcq7TM53oH1jOeMzMghUBBC4QQaBjfUBb6KC/I319B+qmrd0qRS3jDHlQBNC"
            b"PHBkcKBUxb8+BRioZYlFF6tmjFzgDHlpYpVMgDvNzH8NI3IN0HGxfG5dMfHEGJPLNSFBa28/w6bcgmmksJRqBeo096caErM+T/9E"
            b"BdomD2DLM+DVnsNsMzRdOd8rVUg1Kb2mvIygI0gt7xWTJRerInLjd9rai6QozB9LLZSmxp0aVhYT5A02Uh8tzscVbIQ3vsgcGpxI"
            b"d5h/XwFF2qOmk92pMcD2+727XVsm/4JG3plwKcI6Fp3aSe8E1hgLQNjQwCDtXzqAQDloG3+iE4XZoKCk8V4KY4I0+CMK8CswehNk"
            b"vg2Mxi+F4W9ruA9JLQfXyoPyDbmIVZnaetQADzLNsJdU41ePZCTouQr95daB3HMq5bIRRt9U0Y9K9OnxNAmeVU5Hx6eVx5sGQyWw"
            b"mfXc4ZNIsWVUmBN8tO0Qk3cr0Xm2aEDRNxXYCU/e4MD4jVr3rBu8H1UQv+poo0DXdOgtrrSvF/hedVTek0P/suoby3178a5OxA+R"
            b"X69y7oaCf4mS/qjI+lnDgCKKmFQMe+yNdbEMxhhAKwTIOK9Q0N7Fmd5arwUNNnBFjZbJlZ5pJCX9JxLy5QTZ1r9bHjle8YzOhz83"
            b"IQwa4Ue6zYSTeAQ804W6R/GumsP3/jrrxIXpQgvkTZk+vGFcrpm95Q/Um+7SOSg9GSpXFR22a/KyY3wKK+NIyGhR/sBUVXqL0SdW"
            b"YvPS4SuJ+LYej6y36A4Hsku6bQaeqe7IRsailbSQExGKWSpudSp+s++bsy93GmTgjDsLPAAKQbDgEOfUMxRECI4ZoN4RYgyNJbMg"
            b"yDApcWCUWPyfsLW4IFJ2w/niXc4PI5zenaYrPdxEaafmSeMy/vgyIvtyh+etyO0rUOk/IOxzJk8xBkkvZkOVFdErU4ux3FnVzf0p"
            b"IOl/x9YisUtIcCF4Z+JLzz1yKKZSZhSlSgvMHDDKgtUWYv2Kg0nyuAKcptIY4v4amvaJr8VjCMXrEnReJZ9D4SYN0us6u121uot0"
            b"90yWXKySygF5hlNeYvbCtqJ9wyXnDLyyD7+G9SiX5DAunMdI8lLapkKVgNo1O05crfN9lmjMOIb1c+FbFd3j6nLUw8Yw3PJxHuMQ"
            b"8BTvcPabFz7O3asV3zD+7V+PYFgq3pFVxevu72co2RQjCsgRLQXBmFFlg0fauZiPpXCMCiZRzNMeI6/BgRFaaI6kZVTimM7Nt4KV"
            b"3yK/dopw7pQyZtJl6pLeUW/TcNWEaCbDxjul4Nd187F0pKXWraBrFzAarRrkXdeiqt279F83ZPP4aV6DbRewcRpC4gjiUdU/HE4O"
            b"zt1dSihunwpZXKzb8eS4pa6uMQ93GYWUMgZwz8x5wQsvO3LtEVsatnV3ot5/W1sQvGlIdL+F+pfrdSu7oGajrhLRqOyFuoM3OAq9"
            b"BRPxgmr4EEksijTHCBnx0bZE/KpDEAQMk8zEwjlg7YjXxEplFScKBWBeBglUSUqJ0toxaTUiwjKtwjehyJYk3Hm2DPNw+kM7w5i9"
            b"IEYMop4xsMVKMhBTIESZuesx34K+DfASay1QWQudeioNTXAuJsr8tgXiJXoaR+13fJyFbkDtshFcxp0RSmWdjGcuQQtZJGlkxCJE"
            b"TAfWZOIUmG9jz8Rl2NcbFkqkTH6nG5wTY6OrUXUt+P8aVY2qb8H/V/u0NUeySGpAragBVa9YDroXc7Hiuk38aSzaVu8S2ChjS94V"
            b"i0BB/DiuY9FgcRo4x6Llz3iIRUMf7V7ELxzmYFAISlqUnNyQs0kIGwdmTfweMU0Qin9gyhkQ8UeeaecTboI7Z8hp92L9fLruxbaU"
            b"GSXjeetitGnXVsSnW3P85e5cX6XdUO8sReUKTBigEi5tpG094KUDfGRq9/lfjdIa2yAQGx5Lnnjao6rYXUjNsmLOycPtDgEgdKLE"
            b"V66ay+MVkrZdXVZnjroUKT57+FBCeHfmitwd+nv2iJI1mw+D88UTyIPGurmcUvPZBwCY9GBL9Jd27dJX6x4ZA/LY9maw3OgTsBln"
            b"1K9XhjavKdocKik6DpUWHd9Z2ruA3E7I2IFrO1ztyND+OZt1q2FGVeV+crtuTbl74l1r3vPUTbZeBl46GnSpjtcU7RDlAUlkjI25"
            b"GSuDkNOWQgjgGEbMBQg+JnFujHY0BCGZkFZqrDyl9hNgie/Ou8NseUKEE6+6F+PMeS+zzWlqR4TENezaC9jYbCRX2ugDmnbdTRav"
            b"FKRJTW/sH1umKUOrf7ckXDppQMdHVlI7WvWkYcWu4RHWrQycFsIhXd1Cur412eqzvr+R3qWcxxnlbIKUmz3gck76XBrS3ZOEf8s0"
            b"aSwVKlcyds3F3sr1mZLHa8skGLw6Vc3vXtYK0T8estHw6xph0UqGaZkdaD07QD07wLeAOa7y8OJMgYUOjoPjSBkjQvxzkd4aoYSw"
            b"UjjuZJAuSBCaaqu5YQ47jSTWBPjcn/Mrxfykkv/KdHI+l5zX5Q+KcrSlQrKKda4XG8xJZ5WnvD7V3K3503VSgCKrLhFvS+QObpHi"
            b"MEK70d0uudwmdlLC4py8+ScttXGnppoCdss5NorqTOeepFBMgWAhbrqMjLWI5DJj7xt5dYNiThqRp6QRuR7lv/v+6CeD227Vy99b"
            b"FS/5jSBrNAgalDYe62Cwh+SdpAyxPljMGMT/YSedVs64EIiMfxQUYoXsA6jP7OR9RXsCvwBOsBvAiYdKnF+SnfiKKNE9Zbh0uUQ4"
            b"ruai9a+k2xDsgRB8DoSgZG1blwGIWsDn/9m7rizXdVw7ofvBAKbBvA/G+Q/hMSiQFClLLrvCaXffVccuQ7JkqyAQ2GGAyOg65qT0"
            b"59ovcGuIb1J9BzWiKcQi2ZakQ7noh5c+jRKKORaq7S9PLPRyYDmoWNWqvFGtxDQWB8mRxa9lqX9voujyVXZHiUhWpa+spEZnyRz/"
            b"V4pfqArhx6+s6vhjhEaH1RiKEC26Qy9gSf8+XN2DWeH3I+4GJS8hLha3WsTrOv6QJCiGQKl4X0AA3KvUrZZUxWo36FgYBxE0jatB"
            b"RoEaJeg/Bu7oU8ol1MbUve6hMuqXwB1DXh1uXeePKZKvzWRGGpdmst6Fx7o8M1bgVQ2Px472vSbgcojxJyVrs0Ks94CzQ0ULyls2"
            b"Hvf7dkfWYGKpMDx/k+bLTpHloJ6Vt0gXwj0C9jj9sqm8BZvCqNkijyE3U2jYuuBjDdG2V13JCDWRf9H5r0rPI7zd93gCjlIy4ICR"
            b"F5Ig5p2WRhBEuDHeW+SEF5RZKqh0xEvgwlAXczcDopixUoSPP+ClfuxrHP0W+lzWi5BoKJx816UQL4hotDbF95Z4L/a8vGv6CQ0K"
            b"efjuJUxOAchtqCI5Q1/zCiQZ20w4OrhJtc0QtUR9n0fgLfYJ/k827Y22Op6/MuCibK+MGH1n+pxtPTxW7fwYBb444Ya4QNNagzKK"
            b"WxoCNhDfWnIFWiqTWAcWXEy6YAxRlhGIGTlQLTU1QZtvxHDAd4t0XsvDW2I7FfK8TEU5tCDExTbOmIvS1q4zKcz8er5PSLX2YGfN"
            b"lJSVEpUjDRrljkdr+Ytd9T+ARM+4KEN+yQm5JB9yIVszsR97M3LsP5SsdJQ5KbHMXt5DViaKw63y/ssGT9JM6G3ExojYt2I5SIXl"
            b"II1BIK6mfvjhK526UKWe31e+3cvvT8Y3yCalrkV1b+JY/NZN6i+QTd4G1qA48UgES0nWxkJYA2FMUYkwwxK888J6iMtY5DwVjjkp"
            b"lWZBO8OJNc7+un4EegRyZmcgZ3m1hXBotbY20nvN3xawLONKGYG6HlxbAU8QWI6tkFXf51KHg7Xg6XRca9aUuK9Z2WWD7bZkT3vK"
            b"VbNUpNKfZ1WbRR7ccJc7CJyL/6e3zlGU0qqfoPa/zeqjSiHlIDAnRVKj/sj4wS+Q0NK6eRbXzAkGuOsXOBETwlMBTXwqJlSyeFbG"
            b"XxsQ2xO8oeaaxnItT1+VxSRPEV/VPP4OWPO1HsTqaPV2WPNpSUyxNCpgH0tcj2nyHcEgBHEqplvqKRdIKRzrYsyYA6BgFcFOU2WF"
            b"Up6wf7kk7lERcPRsulQ057Sd3pscZtBd+hGlvl6HfBu7gi5Eo7oSXJWAqeDHAViXLR+hpVnZDaVy0VUXByB0n9NkXsQlJdYdoXC0"
            b"kkLPrAr4clMT/Ght3t3VRPHAEggOBtr8AE4TaL/LqKn7VXuaWaxIKfK0eetNAAauit398fpKy7HeoRkjvvX6SmX7WrWWS8Y4PDqB"
            b"q5X28KcIfm0RTJWjykhiNHEesCaBhLjq8lwoFxeHjHNPDDdChICFVlZTSUisnBnWnvlvxWlcaQ3jRwiOR/CNp+Zog14urpLBQKGh"
            b"7+VWxfChl/sMgmRyG7jWmj4KZVSpvZdpyzIZeG8gi6GC57F/XE2+Dv3btMNW/m3Y5j0qvzWQiqPoW3r3mEcFa7AU3aiuBMinZYju"
            b"kkXmCm/sROHt3O5v1g8+G72tnn17FdxwrF/ApP4O3MQ0+c7Hb+9ER5yXvhCkNyYwz2UwQKwDB9JoTrEBE7OxsUxIxZhElhLEmdMS"
            b"tJFpRqeR/ddAwIMyEe5ghPP6jqx9Ut5yMRqKxZJ996aq/CqoVyygXlrx5uQc13sdfZz2mDInwrJjvI0A0Ndg0ngJeAAVZhtU+GkQ"
            b"cEYh3uvDymZ4hSuf0jECmK1ZkTes6Eo2vnzYFS+6ko1fxCLy2OfwbMcefBDAtxHAVMjgLWipvcDUKES1jms8YWnMZshbBUzE14MQ"
            b"lAljvKEWI8MRIkmHjfwqsbWzcvNkxf9oud8mv7II3Oqo3o+hGulvky+4Ry2TC16JrwGqHuO3U6cUJfEyZl/K9W3e3721zMljy351"
            b"vFgv5PNR33gdDls3lD+WVctRmdrWGYGIUQ2eDjVHAhKbz9w6k2oSe4mok+B7ZddzlbLnwOaZPJXgKVYlO4+sfSZPJXhkk/s2PfWG"
            b"GdFLAP9+8bOy4O6y5Nslzg6yZlQprbFxKi6yKQ9gsNJWEafitWE9Z8TjWP5hLpOKQ1xbg3MavLfW4aT6+w6Y1SFPnSOmDjTN64ze"
            b"55QjT2f/jXr5sbybCyzKC2ppSy14XOFvLIHGhF5MJjn3OBN4F6y5YkZ52RGjHG0+tTPzz+PnUuMOGuOjnX87ZISL7Z0YRTVKa9v6"
            b"7F3T25XtGI2fxLr1VqOXTjQb2HKm6JbxexO+dZPxO6tTzxi/suGttRw2qNqm0LRQ8dIeXedUMxXhoxFcu/23CFZ+HdDVyJ/N2L5L"
            b"+XuB7ftWqNdl6gM1CLhJRssCM2rjD62cjiv+QEFbgoyjAhGqOAOlpCOANAAwTRXRQYrnbeTYN67eLyo3XKpTOxsNOXrDkT8cPq60"
            b"y90l/hAtn4B2rdGcACRZ2Vy7pjm0oTkkFV4EUD0lw0tS7IJhISVjXEmgLwLsFUdnu8WVxL6kvp03d7xjihwGkleiDKI2vmh2nMgY"
            b"+Qjiz7XtAAshA6pc3txl0r5L/NgpY3QVvMYmI19Mt/ADQ7shvHAY8CayAO0Ii9fTqRpxQKocThqg7Yah3dLw7OHKZ8AX5ND+pNbC"
            b"zykqrHk1IOxpAEowiwtWZCQFFJMn9p5YqrQ3xnPKiXfIcA/IM06kQQizwB1Wf8/I/nHuPArYdAd604OzVPrppzyuwfscmv9s4/oF"
            b"DXAGuAdvLcVxGTEtxhdq5DOUd1giMaUcDdAG+ISWtX4Ow04HXnaZfo6GWD0HIUWlPWPAVNUCCj0dotswh5dPMRasRDSYBb5t0iXt"
            b"5YYa14RkhETALdpsWY9RIot78nIZbeHNjSbtskRStqhLnB1KOegS+rSd/V21S6iHYdCOyaDO2NDqWS6BVT3Nq7I7d3857IIQeG+g"
            b"dOZyR5+jRSPisi/SFv/xuv9ur/t4dyBBCByYp0IJJomnDktBKOZc+ATs5Ro5YdLUDWSCmSmPOZBYdKcL/c+psd1pylyTOXu+d9MP"
            b"nc76J/hW/2R8iyiKDNlgjjZwtKo076U0Ze8C3Rr2XWgVDRc4l5szOAf1JlHj0HxaZSFxUJzYFxP1FjmsfNo8SUY1H8r22Y/PlMpl"
            b"Gw6CV95+O8aOtRdKOokS+yXZtbucjTG8jE3r90UZc1sPtK9AZQ4CFWdj7aGwrQfTsDNqHfqNvdFymXH16KO39hN6a0CCkZpIC0R7"
            b"LxExhmCN4/ohuZQSknDGUiZeR3CGmNSvDwSbYL3QYj6N/BtGeoPuxt5WXgbEt53yrhs0rbJspMgriLWls3ZMWq2J0l5Jx4Gh6rCw"
            b"nZHWHHCJSjUWZhVsuVzQpF3K5Ij0Uw4UGCi5bQM4aSyRe+aEeQ/lqDDja/u/FZNoEnuKybcrjll9dOW2IdpYXKwInwazxRULR/Le"
            b"/JJUzh/d00VsfivI22drVxyqLNwSoY8KFP3ksn02UstcavJLXfGPyd7dngwwZz1WzmArjIwJlVDppRPMQaAq0Tokd0YYihTGPmBF"
            b"nbdSWBxUkpn4pkz7EizxNAH36sM7sqMwxZp3nZp87G+xYRFigGTrnlhlBoIf2XmOujuSbY5MYs06a85Zzv7cCeSopZB3lYyhscIV"
            b"yqTssvM1GZOv2zFk2k22C6FEDHLdpWlzb0GSdpXXSAumo1kiXe5mMH7Lx25ML8ZT39ASJSv2mqza07gqb3HT1N7Uz8YPXyCi85KE"
            b"d8MPqU9r8YvCYuh6lN/4RXDdNZsZBDouohhJDWZwGCsmqPXOa4t5MhN1FlmgWCBqrGYmlpmM2FhQBiGD+1aS8IHs+0B/7DHllw51"
            b"twZZ4CjB8BX/o95z+Gi4lKqJsV75ZUXLjgpRXc1HMYKK3kHITsxVtQjl0U2IZFEbhSvq77IiHqrQ5NB8QAONyJqHN/AfPTTpKw+8"
            b"vqW8dYkpaa8XcZgALg3rHDqSx6w2bEEnpJg8gdpUJeoToVOxyxS+dRBuip3dax6cLffP2gozWYeZCNpRxqG16ujbA8eGQ91Q+O3i"
            b"Z62f0qxQveFw9y5ZtOuNA+c4RrFqVRQR5nCQnhmgydqOBaaM8B5ZZogIsa5lsaJlnsl4KzBCe4v8R7XynhzlgKA2MHg74dul6Ht8"
            b"uwNN7oK2Q3o1tTpUr3HcRamFS0dhUcvJCI2V69FRfUHkWyutcv+Mff161Y2igJFJerUQ21J9tQcq1BL0rC7lHaE0UiVa0lGN56/M"
            b"jJ7PXhmLCcPgUVtXfxQpX0eJY1hKYwjBwA13RIBD4JHghnEvLePW+VSOSKMZ55iLIEAIjTCTgRgSHlDiBpyRDJo6I4yM0+8894p5"
            b"7sWPcu/9or1cdbv+S9vKzYlMrj4TYpRJcIm4JPmwij5cSE91M4OsTYqj+PqhtJ/V9bxEkSozbemernvu0TKopEmB+XAjfmXEN7xR"
            b"ie1OJdaKe5wqk0NoCllbDTfzJL7XY5j53s+7D7MqGE9r3Vowp0uZvzMplop0INOwpL8RO+RR+lvZHqhKbbjii8xYI3US3NghjDCm"
            b"A1BOEbeWOIeF0STEMlNpRpVWljujlERecp1SHtPa2cCD8ZLaf9z0/mKFeVIO4Qv66RfsRId54Ly8wveV1s91yKpr9EGt2hIHi5YX"
            b"2Vs20zN/rFkmDsewSJGlypWoziBOHqF/KSgdDIWmM1LeYDmibhNQu37ZKzzvb0yz6Dq82gh39dPNgVmuw6z9iSzQso2d1z4rLA9e"
            b"odtwMyVb+Hf1RKt+fGB/FB+kq/y8j+n9M6UpCIYw+BAXap5SLkhc7RtswUnAWmAjk3S6Ji6m8li7KuYsSLBWgUACuz8HJ7sA/roK"
            b"ibqBTLtl7HnVf3TSFP6a8yapdHYXDPNKQD8cQekIp7hkcKkmjI17HnXiYBW3DvOuWXsu9hzp2xVLBn7oX/olb8+yRXrDJLnKG9VK"
            b"vNYbg28vf2Jlk6+hy+6n/4VdvedsXttj1FZzpDOem9s55xuD3EUo2mcllcNG9IM+vVcPD2DkBnT8Mfn8adAZY0Ib76izyMcnHJDQ"
            b"iksDiBKqpI11P6HB+nhlx/8BNhoFqowQVBEh4F+kp1z1sLsILhtQwofTyLMb0EVPvO5ERjPBFeowhCjPWekjCffzO9HDvnnNvDmM"
            b"C/NOMyB7n+jhU9n32URvv2sN3iUF5iOKqRugKvXzVlMcNyHLBglPzPcvgY94kOlrKnFPklK4SABQDrdYKXR3RmqenDNWFrGOXZyj"
            b"k32TjRxH++xoRrpv0z8cxzYElT8k4fHPsVJYSkXEeKuDskZTrgIPBKgCLpO2GzMYk7hqkMyBZ9RoZS0XMSdp7DSodywjGpmjtbNy"
            b"XGOcLzAerC4eF/lXrZkO4pI3rJnacnopVrtC+gpB/VBr08Kq2P345H22Td5bhktvVPGdrsEPpzPy+ThZrpzQIo+Hgpe9pp+bD19G"
            b"HJ69Qw7OB38EMo/Zlzks8ym/whZfC/s7+I8zISW2FNl1NuaNKudcSGl7FeoX1/0ubZ2h/uaKEtk7PN2TXydC99iTaZOge+jJ9NaK"
            b"/TwhC8d10D6PGJ2UyBiQinMV8zTj1gRCYgWvpBaOeEMECJCCSAgIkEHs48l02UTpsgD7NZeni3ZLl9XvKRQBfMRXpc2tnJ9I4Aua"
            b"o3fIBqvaP23wdU35LJ6xZHYoo4V1IFPRuVsPvfhClqKnCKFlgxPxfFgCR5Z7j5Ae7zZhGlstzVDSbPGfxpX/dGvCVBHEWw3kDoc3"
            b"xultokrssfXHR3/+Cf15Jg2IwCSSWIJiUivNuSSIUkc1lcg7QMTH/7zmmBJmXFzyWUmsBnwCr3u1BDJMsNbbn/G5eN6YDbJwQeSY"
            b"DpLLvhq20OPf0nZk6DrR5IYcRfq/80eqTEMRvLSXrHIs9iMbOyOLHKOKvx0taXG44D9k+qGi37X7EF31QnHJ4HTP3+Swy3RsObCR"
            b"N74pGvcyfMbslWeAbCt6+OzR75GBa9b3EyHjlgJyQdTtpULGzGLs06zPxasQe2u98SkLMSI88nGtzhGOVaNXJlhqXFzQY2aYZYjH"
            b"Ff18wf5Raf/1Ku2KlnUfQmSP3677kVDRJf31Kcp5tEe0BL5fq51h8n6ldmiapVDRJMY6bSMLio82+9dTmgJiBGeBKqEYijnMMBMU"
            b"NTaTdCnDwkjGFNCgaUx4RrC44LU+KOUD/Rt6lNclHR9gDp4CPBxFD/g80cxED3gljkMfix6QA91rJnqwduNaTbALgI9Lapqzfqe8"
            b"qnqAq4q76RQWJQNBV0f3SlKht/ApafIrbcX/de2DjyblM/oHzFnluCUxqcaHsRhMqF0raOAQ/4eDYk5rCBiQ1UQZLBG1TFMuvRYQ"
            b"9PO5dYTXvdcdPG8NPiPYPtGiPM29M8GZa+npPzYaVzSVUhmDHFS3OsktUTKIkFWg2GvA+hxSzNKxi0l/KezZIi5DhhyzA7JrqwDZ"
            b"IyWHMRJiCP/a73jdZzkYNYnZoeZTKlOd55qA+eivJ9EZl1YuPbx9zVulwQnDQU47h42M+lhRvRNXfxme6vfNzX9oOr5mzECRsVLx"
            b"uJJmQQXHFYRgEdGxOmUUG4MoVbFkdcBwkIh7BQITETCNBSv629pcz+kkjFPAfA7zjPLvUI+my05crlkHWm0AORuul/FGWoDX9dvw"
            b"SFNUuj0ItDIXdkehsSTZRfEHgfYBC5O7nkPZ/TRpph3nE6UKqebuQVdZh+E55/DyVjGDrnrtmwfHoPtbwjD/qmzXPRvKFsBaP2Ol"
            b"jq1c1OpnLIdVhW73lBXA61bpts+aOXg1hVnBr2wDxNZyCCvE9aPc9bo8zBFxsXQVXAhnYoLVMmZkBxoTRYNjGrDwWgqlWLyUpWCS"
            b"2sTBjVkZwD/g3uIZ+5ZP+bdf4qd9Lwv3QhJvZHOG0lkHB7jmJO5J5ow0aPHIeP0VFsdjFZgrqgtHHRix9UZV9UGcCPPI3EyusjBb"
            b"FxG9Mlna5aJwwBivd9+ZL/feTDkaMqgJ4+EHeuzx5tD0+T8rkVAQy3dkEsbNg9XMAiox3P2V8ex84wXzmhjMa6GE3eGiFRbrfrsp"
            b"pONfREt7WsPmS4IJz/vJ0fyYjr3lONEsIEeVZIxqJrGQ2lmIlTO1mhKgBjmRrnrpJUrJ3WAXiBEsEKyt/Rc5BneQ9ldr66u8hZfz"
            b"DK4yHG5SBr6XBJBN4YDIdpsxdqAExu//Bm0gfYolnFGCaaPQNtyghNEcD2KNlytcgLahydEof+tPkgzyKcmbtLNnDSxklexlNW3j"
            b"tRkd753pxpSzI7lgxj9+MeXsQy14hlrAAQRxTGnhBSJS8CAs40wow8GCU54x62KM4wg81ybV8pwpE+t7h7T9VZbLq0L5fS7CU4am"
            b"B3flI2LoMs/gjv/olab2wVdph6+Sx25BYn1b0q41sqkSAtw5Th92e5dmkFR6cKZCgNyhqNunsOs2dEeUd583wnFBiUa3l3aDHJUn"
            b"gmxhIpe6XA2j05mWwHiPBrUOEMeoixxBO9TrXXDYPezrHMc6q+zZ1Db0HBULB+zr/rsdGbv/7s8bh57yDi4YhL6rcncCeYel555A"
            b"LOA9CiQ4qn0sz6UglgstLaceeQ9WC0qDDcgqy3lM8I7/L7dbHvdSrvZJrrY/7mkQN6I8E/nIxDaoFXDWs+giFV4AY1tCFFU/ZEhK"
            b"WNV4xjqSAzreVTXJ2z0gltEISx9oGyasn14jO9ypEC16bwMtyhGd8OuKlLcMR88Mm2eZV04V1YoxdD/lrFjBc/vn6uHR/vnTZ3lp"
            b"tvbBaLBCEaSVCN5Zw53HWiMptDXxBYN5UqM0KGZwqan0Hsd/tffx138D1vG8TxB/j08Qv+8TxO/5BK1X2g3IHHkVZI7chMo8BZlr"
            b"jYLYWEZh4BN0r8Pxgcp9UB93p41B+ZhJHeaxAE7cCeyVNIwE7RXWmEP8g8UQlLZBxQxKJcQ0yyAu5LTwiv05ObWrWmU3TDyfcYMe"
            b"Kg/cM9u87+U5geXRi8g8ehecR3cz5hZ0nA84fU+gKEB16LxaIBxa5Sk038EYpjvlrt3qcHcsLZvSJxc1Ca3ve6RPPAceza2Pey4x"
            b"5Tg4CEYGtp6dyEL6kErol9TTCCfxL/LGRHKTuGz0Lle6CNSSZ9UTfKKvg0/0dZoJZPWwKC4sM8hNO61mxO16xhVLbpHc/Oim/YRu"
            b"mkDWahFkEkNDFlHhpFLGCGIYd9w7p0y8UwTHYoSRSmitUCJJackAMf9RRH690vFF/WK5I+sol3swWWt5dMSJtCoOe33cn1KvV7k1"
            b"YYZndsUILr1lfme2v/My0Og/zhSTbZMk0Ea4cnN0asJTFKR/xLrSUE1bZbCByKcGCLPBxzb65tK1mD/lr4kip0vpHk6wFjwjFYdv"
            b"Vnqf2SOV0l1WVnbyiArshHKg81quuIAvsLf7KB8f8zGjGDluiAmUBmOCiJcrE0YIJAxYYak0yMjALPdgCPOAPAUiYhFrBXmLjuUX"
            b"/TtJm7wmvZH7UjjDtsjRN+myEM5QARhfS7sbxFittUUHz+NLRiWMyLohsha2R2OkWaY+YMfLPuNPSgYtlMG+p/2Ptpa9sOTJh1fe"
            b"+Z4K2Rm3sNtgSC+8nnLT1/+/k3J/SW/kDao470u4xomgsVaaenCCC8OTbDDzmCuwThKMOBgV610iKfcceQHCMZFCrSL/hIdyy4Jb"
            b"bY7wpga7n8XqTrQsuYnoq7neiSMtKeXA7qhOYXIBU6RMvUQuyyjc3hVEiaExoR4wCDfr0bSLUmSWweDumnnQvHzsFj1jA52bKtXW"
            b"yt175qNKsY1v8h08xJ328FHSZiP0jcVyllFbJ6qzj9RGYjlyW/KPHxWIxG/nmIwy29uZJId+rnAYMNNaAmeYW22I9ATzWDgGE6SJ"
            b"b47jBWSxSdbwgjkrKGbOx0KRgr46IPvlReIYrEuvqMQ8FFWsQcxfF12flpsDaZoBJ2KABNhawOMlfvaYq0kdw/2OeCvDBnARSSzI"
            b"ApCy0zwceGlkLkYOxZSUNussVRemSI5qsGITy/sxWuxteXGW/eY57kH2q6zoa9p0mwO/nis/ZeD9MtApaSUiygDQIF0asQZOtUkm"
            b"WgYpq5gN8V+MFXJMIa0IcOklVZpI7t5QBtZ/iF0i3YkJ8z5nUr553lt4mbV0RcyFwmoA3hotyo81VXPMd8b5AySFqHx/yRHIEP+t"
            b"UAzbqp32wbCgJDCulRvU+gcielhByaeoXgWrwweUzyZFgeTi2N5s8REpJA8nF5f2/6pu7PJ5dVukyHK8mC90i/p76JoZJSYn+K2x"
            b"We5TbwFzrdplUAlGQEWDG6mSLdUlr13ieUe4wxXhDh9HVBWLbvxwXXavw6h3A7q2XElgg2UudajYDJ1uALrKxTjvd+6JdgznQm9F"
            b"H0hCtQfjqMCpQmXANLLBGyOZjnmXkphJORUhLqi1sPFCNIpiFAva1Nqn+ltHTF9Ey86oawcPzQI9SD9leyM4KIWVmNwvU7xeO642"
            b"Xn20Kjkac4U6EtgGxt4rywJ+SDbPpD1xsTfEt12nHaZbABZ4+ZyWqm7ZfdudlUuDNt4/OdprgwZZSmdIY7mzBXpucd5duVkxtJAR"
            b"qj1T3B1IDsppinK0nyZrM3+nGFRC40+oJmusxejR3sShMDfWwns2Wysh5TMkChhdNYyWhcz2RqT7fHJshjKDFP1RbQDlbpMYWc5i"
            b"AyTcROveVBSqyW4t8W2uNTRH+M61hnbs2E6j2xjQs4cdcPdVjYefRfA+OeB6l+X8ZeSBJMJSqrzWkuKALRMyOAss4ICUjKW1EIJ6"
            b"g7UWylAVpGKKMZZGXURa9m8jDy5iCi5aNr8YeXBlGTB27cG1p+ZgJLXNgugQlDugx1W3zaus74PkUxnNVdJH53CH1WA0XUKI4wZB"
            b"h6uN+8PJwVkQiRKK209llSrt+H0pbtGP+wrugFEuRVy73nRjftpCbW7LI6crAFkb9sxM1hpK9Igf/UEjvLQrIpnBzBLHeCDOI5tk"
            b"io132CPunOTSgTZKO41T20QwF7QOjhocDxCDeGQZMWTNzThzpzn7F7Llhv3Lzplmvy+MeGIxiEnUOBXNog9dlylf7tjpGd8J6Nqe"
            b"Xroqa8d3bSK0hJJ0nDmSCilbDllv+5Mj4s2frC4+Sw0OBwO5sZNQuz+ZNeBIusXBTgE8mlLn19dh2s0KuKyR7vSNx2rteyJcGxv7"
            b"kzWD1mb3zZNdXqJ50jrW7xS2o8rmrylwxfItHrx0lhzYiblfKlhXahmqkhuuyGkzilqdBjf6mTQBBaICVy4mv7hsIgQ7EQTDMQWK"
            b"1BVmyEstcAAV4jJRI+OQRcTzgI2iP+aT8y6m2j0Tip4Z1ZlQQEtgaw1pZc6OvDRnl6NhBy5GJguh/wSC3pGnLdMQ5BSydh9qXGzH"
            b"soiHl6IO5hNQL/h7q4isFYHl6l+2b9KL2pSwmHnpqfjNtVll3k0OqP0nbvRrKU4bvyabjb0nllzWpKgaUpWfbgUKP9LGRrXeCoKS"
            b"L/Gg+DlzsG+0AFvTmUVGMemRNMZ4pMFT6bgQ0jkb19gaac0IIfH3xAuFLQ865jPCKfeOcvHxXIRlDIVRVYzBMsKswtLrMv0jGy+x"
            b"cxbrmJ97zW3xmnVjAWnyDRKwdyWPiS3HpCJtQVhtLovjvDlAUTWujAcJiPTZLI0IilD7MR3t1FJIRj9QvqOm+EQyrUQtGpiKdBZs"
            b"vZhlOt4S96zQ+l2K1rz2W1qhCzS0fTyv/FaCFq8FyCrSlxwQsSqJyOphNSlbeLj8Y734WutFxRxHjHpLVCxdLNbgwACRxMXfqph3"
            b"Gah4UFYoBxaQVz4QK60gjFA19zX7o73OCwj9YUaYNUXFQT/mjLwkbusJX+uyXiZXvY8LdbF/e7OPmt97IUxtnZsFtLFRpg7fKd58"
            b"zTlFtDkiGOAQUoxET2vKsFu9TVJVq6SpXEk1fiLNKGo+voLVTHcD+ZOHthcwsLpoI3/7QKpOv7UUwpcEZd7UxlQ2VrzUeg0cey6Y"
            b"lkZxgSw4bCVXyGGfzNaQCVRqrSURHkslPQfn2Vys8W+4Xtw0EOroODfnPlN1g8k6+qIi8FhivdM0ON4eTswr4r569YPjHo/SByuz"
            b"iR4+HL454tKtuTtzqcj6O7nFStQqWtDjCw66BTk28adQfzCs90fKQc/6Wdxtdz6jD7PYUS4pE6qUCYtC11rTth5C0EnoVqOhWjj3"
            b"pfpcH+uKFs6lYymgsPEMbHLilUnS1lvrjARBsXKIK4KxY9xqFOteYSSSgjPCVRBMwf+y8u003x0NIBZIK+Od9/kBxJVj0tvKhCgb"
            b"WZR38VnzpsBaYdt92xo9wKFKaJrpVYZnfGLxOdZkxPcUgNPm5f0wFmRrXPDDO+PehbPVV6xoCs0hyK2pAzVGtlwtzR7TvnIYcNK1"
            b"LPo3L4ea475L+faWLTqpWAxkwG44Vrfn5udzK3Wy5p3t0S74sv+ul4P5CN++WkpRE2KdNsQaxDIZDAUpgGsUsKGWOcVjaRwYF4TG"
            b"QlcHjZAIgQdLkAEEr5VSfAlH7KS/OyUY4JXXKc5ZXxdIm3hEeLqiDoZKEavqIlYNDctfxpjNLxfkKyat2lX3tioXriSbmpNz7i8p"
            b"MevAfQ3EaD5vfytH6y9wV38xH+sbWVdrSgKmeXCOCkLAB6BKAhaB4iRCFX/peVymC2whJHM0LUEqpRAHgRzVjnxMe+er2jkG/XlP"
            b"3Lyz3DFUW1+TLQvP4bz7ik4rWhb0S8R69gNh1fy2KfCy3e/HxLfDWnZg+leY+I7RmZ2Jbwu6/wi6vnINri3mRnvMOYk5UQWKYnVH"
            b"HCMySCow8ZZKZi0ocAJ7pzyJT+MhIueRwp+G5ssNgY+6q/VndSa6emw73mui3mlSXm+R3mpnlv5kppGyTSB8hbZPb1c5NI/q3t/U"
            b"vCsDNTdlZCeOYYuA0yb4BJXgE5SkuTU2ZSUfNXbfPcpDbVmVfRqbr0+qBlsiLDCOXUyoFikbhI2pNlkMSBnXytJ56ZhWwhOQOP5n"
            b"mGUBcau1wu5fVIJ6nNOOI+dFAu9R1qH9BHghdZIFcC4qC0LR9e5QiVqzP9nmJ+3yN6fQvPzdirZxfmElIv4QjYRduRgbxbsUkiPV"
            b"etmrUdwVCaq0jyw9jqRoe5DNzU6K0qx8RgcqkT7vNAhn6QtP01cpHinZhaLrZ3ilAuEaAb9vCRMf20ZQ+ovD8P8dWShDkokVQkSB"
            b"4dp67rHliNK4drY4lolUMRwAWRuvKcuBcwFgGUMSE+UD/0jZnUjZXSsVRZG2FAvPpFX2eJsy3op7VBW5/dCizL3AbPt3mjTzseeY"
            b"mHnoQG+gzU00o2Pflr9uAiyHrcC3vPLg0SdfXcpXTAnnwHvgKf8kVzFhAnEErFUSS8oMkCReoFhc2yrjlU9CnDaufUHPGeA/4so3"
            b"Q0qiU5EmdibSdOAWlrGq3EuUfrycXyVM1cmnZ9W11dqKm28yncrNMboMXRc99Q1h/4Cnjg4HnvaUAkCpbozClpn2hTl5KSn6Ht4m"
            b"oYcGqnQHEFGOiz9gJKNygIinuJTxKVP8eNw9XTOeWw7cOIY3ZZT4PQPTsb3dYjS69QJJJUU8FykmVWewfjw3MK0lin+9UlLDKlxX"
            b"oBcVkZ6fyh4msUYRSFhvp701oJXSzHvKEEbAiQ+JY81AIJ1IOhLF1GeQCBoxSYxhhLxDs/Ot3JqxPueRnnKxm3fm1XTqH9rJecrb"
            b"ip7XO4CXxy+XpTcPXThSdeH4oBGXpYWyXOeqxn5G90lHsNpLKQLt0Wwf3MGjKYWmA2IIteqemGzg7o4fniPzWbC47OXNRkNqUTmD"
            b"EhxXsQj3yqDHK6mExUuu8V66M3a5kYJn42U2HSKfvTLf21pZ1il49ruhcvz3+Up/9wjmCLtpdDO+fzgzQJ4bFZARcWEIlibtUBXX"
            b"24FKH3MKgVj5UqsFcQiCVcAYBuNDLCAU8k5I90XnvZ/I+jfS9HNiyvSMB/ngVnPnxnA1658Z9I15ivcQ8qtp3bWEfvnGUo42nxqr"
            b"SVd0qcWqQc+RlckyapEsKwS8ADNPeKI57+e3Yem6P9bq481EPrq0wfWbRjym7abxzE0gX22vYMfPhfDkgHH0WAgPn5lTjyfvnfLd"
            b"P2yv+hduBZYyKq0zGisZrBWSGQ5ghJfKhIA8ccnOWhqMhTNpAIW5F0JxTIwnQv0J3NMLy+mfgVBVWnC83t90i0nGvVPHXyInHfeZ"
            b"V7WJQ3QJwJRRAuW+s4vWLXJ3W7Hfr7kEKnunlCxvwRulEzYCP3Fc4p8EP6Vv6q5Gf41Vf6xOMsdLzTohc/znCY9zefjBPb1wRG/B"
            b"xdzphDBaeqw9SKQZMs4HRwxw6hGYWAQQR01MokECRzxQA4LzeKzu3+bRv1gM9KjH/Kii3oj2dFT1DtxOVpX+c7cTXAEtr7mdDPYL"
            b"m4UJZbzxHRjvOIcV55NKTWS5H/RJL4dk0W2KKq7AUf+uvJ4ZW70zysHe8MwZBVXuDzM20U0Gfbp67hnl1Sp0sjHKm78yxqOWVxqd"
            b"vGZvY6O8FgnVoaZ+m+rdP8CitwIhhxUyxmBAznEpjQyYcGACTEKjUkGws1TLgBizhhsWYsYmwQfD0WsL2J6JeU7DfE9deaXafQ0f"
            b"vm/BXio6rxaSRwDpDn8ijyppsqv59xTWecV7aBod6fA1JZW21NBTOvzGDn2AH32Ga3nXvnnGjD/zEp0x49mUGf8uAOmvp1zm3Enr"
            b"8vQO4fJdFWowmAL3CDFhkDOWc+EEocBoTATGSm+U9d5IGxf6hmFjRUCggkSCGfqFJFlavfKkypyT2fFZCmVXU+gzrMhDlq1PY4ZC"
            b"xQcR0jGyfrhmHqfjIUP7QWKEw6HO0iMM26iwitvVFXFZwm9zNHl+d1CrHvMhnRbt5mJ3ItYid7zbfAQ5dpxOLyTSrEV1I5Gye6z1"
            b"M276aL2Op/wmPOU3nXlDHR9uiRS/hN8k/69TuSakJFK6qheSOpHiPpGyK4m0bpbOEumCLtz7pmWZJPtEmu/ib02kLiZMw3nyOFUK"
            b"ZUEmTUOIRafkzIVEZdeKSCy84kQgajzjFDmvXVztW/UPukY9x1YSK6tTPhzTjWU4rzsb7aMwXK+xp7pyEzLSZLDY8wy2ORU7yMkN"
            b"uJ24KrmP+ISL3n3lMxDZwgO1MLuxIF/5JPJnzMeos6NJSNxvCeeEENpCdSdyMTkwnyxntKgKbmA8PBnklcj04T/rDFUuojvkgr0X"
            b"AAMl041BUD3B60yM10Ox5rUhE6v29GsfTdJ6D+NdfADX38LHIOrbDaIcV0FzQqkzUpNYpmgeiMJKSM0sFRIR55RWzioVgqbcs6TD"
            b"b1NNzTALf0Zoqi48XwW+u1J6dxpMA4WickJ7Z7U5i69L+I2RCVcNtB/4Yg8VohlZFaIlqrbg60ZHpeg0Ul4a11WLeU+sPWo4Z698"
            b"JVU9o/UuTEbm1/lYMGO82n9zXIMvJkdDLtgxrkEYGM1MvHNkduZ+ci53M9vPmh+rZytUnq31fQBXLQ7cze9GfNxZsV6pWne5ntdW"
            b"r39bmeBLTtpvkpxyGjEbGCgLYIGi+EdqhXXgfCAaK2awjaW7FJoy7oRxklOjrY9lveLW/im714uWpRcK94tLgBt1++loTj7uJ3d3"
            b"u0eA4TPju6uLkusrjZE6webmOsBF0xUrsfhJqeaAayPYMdo5b3cEoe3bXQNhi9Fk8hzp3Fm9z5DOty2tbmMsZkpaY/WtMyWt0709"
            b"0Ni6isX4VOvfXq17wiQyGBnOLUhFtMaCcqWFCEEJY3nwLjAlCATvqQ5c0sRvYbFYpzT8L8vCXpJDfR5Yxx/bLZSqWfR0wsfl/xxS"
            b"fTzNO6yXs8qeH9h3j4r0vvFU6CsLym+v08VW0R89XwVZ1w8UNze+bdPJkaFlC0yJ2rZbHHLIWnSoAfBDfVEZlguMYlEFNzL9mQfh"
            b"rHuzuBPi2qOmViDrbGN7D9kO07yZ1TQPayDzN1Xt/4Nasc5LoPGSCw5xGwSSlDGvlbUBS4qF4hJJG0jM8YIqFEwS9A7KcAfEOib+"
            b"bXDdFTTcNQDeZczcc8uGqbP3T+PvLi4xCuaudGpAyr1a324V6khiLKG/DlP3o4xC3Aw98YYjOWcUjh49krf9QOnuQulcIDGDEmqJ"
            b"84ahwIURFgHTGnmiCcJaeaEBSZa0xriUDGRc80EwOIbqv0cGfxnXjy8ZlmBMjkV3bzMwwfC1oojp0EtLhcaVO6+8Z7e9NsjA9M4l"
            b"kjKE5MGEtjMFKEGlcxOXOoqMG8+9PNAiS7SqKVbIgC60JFauKj/O4eoAshMvRoIMRIe6fWZ7iIUtQvlROrwNT28OI/QNHWMN8y4x"
            b"xH/XBLxiTTbtjsFHkuPLB8gZI6zZdvjFljCMnieEw30jnFGPeul68Bq4wi8wBc+YKmRgjbtX1KMeyV5l7w5k5dGHDfjtbECPKKYO"
            b"GcCKJhwgI8iBpjqmiXilJzqLRBCfaNAMkSC8jAcqtMfWMm3k//Ad4LIex3Uq+sUcSirfRFQOj2/gjwMnJMVkiMtCnqMbR3oINhyB"
            b"Z/BuNfwVkZNj8V/fTQ6T2nwAmddOVsObBa8ym6yWyHQwoDZ6TWOWMXiXFJiPiBEAWLE061bjD6l8mGUDnlhd+xfG14Uk6+5MJe5Z"
            b"OjiVkiN5C+VC6/EmJRdxLpTUz9tn6zSz7p3sz2a4lmoCOh+GrnjGta3yr0xG/+59wQjKMRapSZ6UogBkCJQYgpiUzlsu4qXsYlFp"
            b"g7JeMG+tQji+iKlm8C/eF66W8ROcOL0IFR9gJl5LVb9zQ7pHG1oIMVn/gjf47YmuximunR4+2gz7zvvGFGO1w76rVcz4rW4LWMld"
            b"TmWB6GzLpKGyFC13nRQcbwuNAsvJYZXTKBs8eWdIwiRS3lPtm/m1F0U/XoNiODRkIQ7VcqF6bQnk6xqhfnaBJbn5vXch5fcn8Z87"
            b"ww/cGSwwrIOWhnHMg7HGooCMVLFG8Sg45CFmIkMToEYIZKhSFAjFNmjK5N/2/Liaii/0zA+VsxiajDQGcqiIQh7dkRYXSaREC1ih"
            b"nS37YueJZDtg3kUVNjdPJHMpSNbGj2iNOdtgQKuVZ2VLVBXcbfxFlHs+mxQFkouBjnXT1k8h+X624FBKIF0N9/Jaod0iRZbjxZxA"
            b"oxw9+uJzTB4KPGn9kS+d6/V7mV5CxVGHCoMIlTc8tB5yw+7OinbEFdoRNzX9CfWoAzZCg0z/OH+8jmvkicOSawoYgzMo5k7PtZGI"
            b"YhGzrsVBcuJi4WYsRTpWDthi7w0jFjFiOD7FrgyRKye4ldPy/Nfgy9dp6BQ6zota9dps3nvXqAVikNyXWM5CrcAI6JEoKSRmutUG"
            b"eBbYZu06C/EH5p+z3snAApS2aHF+/HRQUY4WmA834lcEr4frArG1coSU674Hn0P6qHLIcxac8jVzyxkJc7NXqjvlVSE8lrYeiVFv"
            b"GZL9PrkluVzuVf5DdaGJ60LzohHniuVAVXbDFRpkhgmpx5Ib3sO7mMhiJoOgIIE8AnMCEc+A+dRyAENinsOaUeWp0eBIUuS3Ongr"
            b"kADxS/Q83ouMG63jJ8bBB0z10IG4I1/zVcIeQ9tqnWplXBz33WogkFJgUXlosoyo6zRXqAONvJkWSDrafIJUIdW0XTrxz944IIUv"
            b"+qecYLrDNXpRvkPDm+Q/MvRVb3UhBLnlsC6bFXj7DJfVOt9ryPoZzmFleZ1/0T3Nr8ud4dg+O6smYXu950OW9/+IgLy8nvSWQiAO"
            b"nLCUUqF1zK4aiUAIdxKDMZZj5wHJZB4nNAREaNyGUyIFCl9q4rKt8fUFKRDxSA2ktxWZdzY7UPHzc8BeXuRMeRkPsbi3xD9Omqob"
            b"V4Q9oc4xkzwRrZTIZbB0Rq8UtY+i3bFaEu/FaedUkpWfU/C4YTo+u/whLA1TEIz0KiF00I+fyE+NUSFlr19CdtyVIeH37KXGSqNn"
            b"K3xSPSMDPdJjAVxz0Hde+pjhuHJemu7AZSOqP6lKUpXPUKOn4Qf1Skat2hCAguAIWec9wp4SolLzVnnpgRGFpJYxEyJsHePCxJcc"
            b"Y5g5poMW/B1DvC9SVJ5X4u/uOC8v2u8wUK4Xz2d4iQGbZL2tDJAeRy5PLoczbFoyUeKr4zh/r7RFeR/MsVBNP2SHW+CqN9GuScqd"
            b"Ov2UR4wJ7Sd3KyiSdlBLuf/RVRXAGjrDjQzvRQ9xI4fJaDrtEt4m/5vFPSVP+EEPGhaz+8K5S9fci6tvaNwBau/eXN+C6/vuuv5h"
            b"9v+Bin+Q/QNR3uhY3zvCgzaUGmGZc8gHQaUzcVkASKC4CFASxYrKBGy1AwYYlGJzH+kfsTUUc3oNfkSv+SKfXrVvcZfwsrglbgMs"
            b"9h8/LgcmGOsGeFx6uQzzw9233R/L1DdGoE5jZLkbd2sgkVvfCu+ot/Vcu0iFy/FhSnGtK73BtlGHkk5x6RiwZKxi77B1WTDEex9A"
            b"9Ts/6giqTzHpayTLiHK/a9Gmm7m5cKfDLzfIWPrI0UHR7iTSsZfgZ5k36Su7jdw+ZHc5zeFnr8yUT2oOe+fi0v1uZ7M/yuO/nc3+"
            b"JfLNm3iOIdbngmtBKdPYK+Q1ChwQMMWIUEA0R04qHWTC4pkgsCcaJNFG4hiFHkz76Dg/z/EV8+T8NBgPPxjrscumV2McWzvba6gp"
            b"TfYSJaGvzZaOcdIOrFagGBV8QDdBgw7KAlo7QQK2OMAGBJjfh1K5qG3kPa2ZfTRQS5E4u+LiKoPRSt9pMI28erPiyz1G8Orr27Jv"
            b"c5MRPH+HAkHn9T0aKwq0r02UIn3qHZ5nuuPk4KfxFvymaTepaIqkoyw2CIvm8XHgeNajWbPs4dHeSpl0VX4NzmI6TxyZeT/IvY+Q"
            b"FvVsES2VMOny8Kh+Jlsexks2pkuFvOZehamhxCsvjQJhHbIgrUHaSeKQYsYKpiQXjEkUsEIKJYiFDfGCDIjg93DMH6mBqFNsxXkS"
            b"fpSB1TCxnfU+n5KBHeXxL5LHB3PE0tVI00xy5AxeI5kv68hHun19FmsQwamnsKjq9XrW+3anyHE0vmks0tfxJ6shdIuRTL/XHJYb"
            b"/Byz+qsqiy7RqfMVYPqzWfduO2NuEz5uWZ/ZhD9oZ0ydalsr8dcZt3wHvO0NSfdtbPOgJFFaBCcx1t5zI3iseZlgyGLPY9oF7zC1"
            b"iktGhZWUggvADPWWK+zlMwXv1mX6OsTtvX61lzLl1pe462R+0aD8qjv5xHd8Sc7rpG3sOZ5fz5lcbm3jGR6kFkjaAB6iM0E8tJY3"
            b"biTZv+chd7HEFPODjet4Qi/Mh5xb0UvDvBw7qb+Ig2wrWj28CCPLe8gVNTjbammvpw2eVEm913eYAeQWBaXqFbK98lxHYmOLd8Tx"
            b"3r/wyCv/A4C6aS5+Xh31fQWwThBj4xRIb43DQkhDFHIOmXh5IUUsAU6dEsI7KRGXhAdvYl3lKGBv39B8mHQevoXWcVzKi9aNNY2T"
            b"yGIQx9ocBB0IIv3p8aUlydYBWIc+yYAnNFo9H9bNKfcqXnc4h0yyeHQpCiNEynUlW3OCg8MXKlpKCMvVg1VW8qX0fjcm7SkHxPsP"
            b"arseB6oKSj/w/t2L9vaV17wLYvhm/Znue/dQw2PjbDY1bmUF79boztVEuPx0+2vk1XaFMrGO1Zonv7fuvJXtvnFRv+Yxo7lwxGoa"
            b"WCwXJXXYAYrLo0SWoFwkXqaTBhC2ysTKwHpPkFPBC02c1t865HqPNv+zTdbBkIvc1ou+os25adFXUNupovwXZPyHNOs7vc+zXgB7"
            b"qh3wZRn/uTJ//VZXlfl5DbM4F+W/22ylWavp7qhrNJyayfLLqSz/ueRRVUpuoDW5w5Sb394T+PztBLcv9QDeNfSyEnnirY51JsMO"
            b"ER8fB4Xilc4tlmnBrwzynjnqEGdBO+UCjoUqZpQI9q2N17fOuL4k1jAxyn5V2/ZEAP8og9eDnx9kTbRkbtl3LkeCeOn9c15WdD0O"
            b"toRv2X4geETz/nHMtLj2rWoHZYfGxMQGbNhmKPvOc20kxWJ0dXyLe+66B+Goo39iucHitWnQNygIXb+JpE0km+NilZNtf0T5JMom"
            b"G5jtduM3fa+vUamTS12xC/LX5lp7kQ6NxjNuvLxqZejqFlDl/fHDzVHx8LBGR/xbrOcHuv7fz4ceNY9dvHMQJQnHEhDSwSEhg0TY"
            b"cIPBO2RIXNQ6h70KOljDrTXg41UpGSMEfrh5/FVl/mEDlj+FmrjWpj20mof46VvN3Avd02ctAiad7JH2/5CYfdTz4bJaDhx6wwNS"
            b"T2nyZtFoWDvPagIeXvq7UNANq/7z3LjnKAD9lG04u4eImPnTzlomaxRUnd82aqxmQQ6uh2NERO+k+HtQxU2yXaGH1zV+7irvv68p"
            b"7EJQgKmXzlniGaJgpFTCU+ljLlXKCqaZ0daBTKZZXigcI4Ry1say698pzi8Uydeq7csj/ZEY3FiX/jvE4Gbwi28Vg2uwcIPPdqAG"
            b"Ny3BH8nB4dtycFvz5boc3LEOvyIHd7cBflMpVJ7owckTPTh5ogcnT/XgmjZL93Apwis9uCak0oMbx3/q8m+vyz2TLAmCBmIw4hyo"
            b"ErEyR7E895SRYJPRVswaijMgRBtqiPHUgJJeUULewTFfG+obWfS2fRZ+SD1/IJszIop3DQa+HoR8jNYd+2K1VuRkzShdKpJoT9tc"
            b"bsG7EhAZFbO8HGgCIcith9KbBcyXFXvHqA+EZZ/xJyW7qtvC0Bvu+yJV/QKZs6xX8jsf0XO4XbrV6LkFESfoY0p4CcO95MdNSni6"
            b"CO6MMWXTKCf/taqdUHVPrsDlSjteLkl/f8z2RknzsAXQHaB0S+Sv54BXKXgZxl+o1t/K9D7LvICI5Sa53EpBVCzBNeXcOsG8D5rH"
            b"2t1KbikwcFQkrh+z1mjmDcGx1CIe/W0hzmvqmU9hlGcTxytCnFfdY45yS+Uyk5MkftHhd0Ys7845v+3iQ7uznXdE8Kw8vbUAyHz0"
            b"rJkUa3RZL2zUbAnA1RKcx4n1gcG8Yk4Sc+k0vja6vJdsn8Z/1JKczTNW7As3wEn9pMm1FTO7ejhOxh+JzldLKsWk65Uk1GPsNKZO"
            b"M069jYnYKiZQCJIybeMFZRSYWAIrh2LBG4xi8U8Ox+BPi+TTIvm0SH64RYLjpZEIjHfMVGYtEnzSIsEnLRJ82iLBe9Pj8HDphlct"
            b"kiakapGM4z8tkm9ukQCKf7gOGx+480qZeP8wUmoM1hkdKEfprmEQjv9HmmglDE+W5Rh5oRQI8g4Zpi9yCskZp5CPOIWXSSkPTLcv"
            b"gPBOuhRP6T1NuOevpSLiEhF/4B3kJyb9k4wHjD+hUTca92VymGynkhXgvA1V5aM5GNgOuI14N7Bdyet1/bLfOosO/VMslrtg7rnL"
            b"91ohV83rqvSe402W16rJJa62mvVCeF1/H91LHkK8f4l3yRv4LO9rhuj4h4soMIecpPGvjgAYxZ0UVBvw0jDqgHjJuUUxvQigXisP"
            b"IGNKdjF3vVRE+iXZ9SS1zgDYZb3fvOupCLRokckxQLJ1T2xNK2r59k7V9Y4KSRN1ajGh97UKzmV3jVA+y3GEVQfIdygge5K1kva3"
            b"6H0QsXaDhxZ+V7rLaR95lfCU2v1rOXtH5MbSUNgq1f2xXDq+uOr4Vslw4PBaPVy6EH8ym31jzlrzlCGeaauV1uDj2kfFoo/JgIAa"
            b"5Lg3GFnGBGXMWYoCN85j4MQJpb0P2L+haVvL4HY5aodmzdXV0truTFpNfbkdK1YhySuNVrnO+0ZZqOC9jsc16O02p3aruXps1XYf"
            b"41TcU1bTjUGq7MU9m1ZqV3YurdSkz0lRre/WNFOPMGCFywFVqp7bNmr9WxQHHcw8HutNlOjB7WRgorSU/5fnYreVkuf6PKNXymNc"
            b"zcQev9JrHV999F41tS0DEthUEJeFuNi9Cq6rqeUl3WAhvipn7sXgWEsNvbctaxPc13ALgQhkMfOEIa0ZZZhpbphgnHKiEZcmZmCr"
            b"KAtSGwECacS8uu+cxKYUwFORy1+ocDnUtzwogTU6jC2iNgUxiRpc9Cz6kI/5jGBxvA+MO66Ub7Lzckk/RTyNjpqO6ThzZG1ctJQD"
            b"nSNUjqCSkBWrvFTCcEBhjFXo2v3JPNTL5GjYcdfHLkB+fa0fb4pN5nvinfw4w/HiZXG8Pq2fLGYf1fCr2o7XFIzmycp33j1CJory"
            b"v0ZjUixf4tE3qbQbR75Jj7QkX+mbBDgucuOKlsRU5q3R3hlJpSVBMx3/xcqY1GTEcQ3MEJKeM5eUJBmm2GsbvuCbtLTnv+DrcW7q"
            b"cd+DY4YQko/gV/RZK6eJyfMlh2e46JE0JGU/ME2Gw3mXvRaVG141KzGpdOH5Q5vToYnIiKLGT3abjyDHHiXRSCtblkdXRefsK0Cq"
            b"fBJ3WGljTJScyo49txIfK+F0TOQpI+0fc9P4Oc+MNZPSuOKWlhkvXGBBUWVkLA2FwQYL45UJ8VCwRZhryxxISUUIAusAJonnvGeo"
            b"/0gSXc4LRvKoYHxULV6glXUL7cs6C8flMXnQO6RLVhsd23AMs1zZ5Jq6Q4rud7uMbRqRhuF+L3Q0YBtS7USyzU7++N51U3RpcW4f"
            b"ABlN1haNiEw/YwAbplXub7NjypqzzNFZoaxXhoRjZXpUhrxZnRJOMMCNXCwbDm/7uMJVdU9m85qtVJVrdbo/mXGBazmIjgvcxP52"
            b"XfS69VlP17+kiv6mQQ42SAmqAifeB6M0B8SwRir+1ioZi1xATlOBIV33AWECnmIuFNPARHB/DNX6kEgwRi6daBUczebOPN/pwfO9"
            b"2LiTlVZwkCnstiksgZT9spd0p8/YH0zeb8rFmMJS6aZPqRVaoPR2gs27Sz/lurZeELpne86HXefZBwdf7nHZlb5hOFDam9aXIxlm"
            b"4/aIrmXjFZV7IIBM1HrXwHdJN0ANXG2erGQyXAGlcFVAj0trOSqnHxTEvWDk3l6QL2ivvgTntGbiqkYuSXnkOVT+tB54iebsWzSk"
            b"Rpim99bIDgVKHHIugAyckphuGTDMsbSWBQsMVACGlOJBBiKc4sI6Tjgwann422yDF/IIrhEXTtGX4jkY6wRnewlkS/sZ12bUXIN7"
            b"h6dS+zSPE22/AS+nTfFq77msWvDo7NPB5kiqimPnctC0Ai8ct8rB6ZDWVitdCm0xkqdn6+LhSYoBIxA3f4JiwBuOAW9IBrxhGfCO"
            b"ZlAZNrdPF6LB7tjcPHuWavDYwflDNbibceOSDgFxwSlJDXggPjBnhKbBWCJk/AfL5JMicYgZIOmjmYAV8zYepgPzb3clrrcIvr99"
            b"cbkrceG2crdxcQmikJ0/E3WY7nFyG/SPJOXv9TAmYjgVjqDhJ9NVC4cSxdoP7ID+TxELQuG5NgQlt9vBx6G/nEIIluqzBQc0FfBo"
            b"b7v52i40Nqxym1fxp/HwwsYDiXdpCtYYRywoS6TxXiHpUyvCIUm0l4baYKSwVnMtNNKIYE8kiU+U/jC7Psyup5ldv5VslY9+IVuB"
            b"WMvxMw+j9DWWUM4YYY1eQlc7LKeew+KV8Syb62Zf+ctkLlyTuXBN5sINmQt/yFz/IpmLKGS1kcRphYTUxgZpkZWxPKdYKuW8B6Q5"
            b"YGupI0SoePsw1pqYYixyzP5jVbm6IoRzKLZPnIzxaZeb7K21J2r3c2dkPJfW2bJxh2fNBsfLXUHinl519CDuzKDX8rz1KV466mRf"
            b"YkzPvLxrJoNJRSqQ76bDL44+yExsjAdVm1LL/Y+1oTSofDAUGhRxeYPliLpNIFt3ytq6k2wAUnQHuqFEvMfdaZvQtdNRMm/3lK1S"
            b"DdsIECppG154D0vSbp8VSZy9idI+y+9LN1+5rRPdmXxUDLDUQ+lJYj9eyz/oV49Avw8S9nIFwrBfnQ/6feU8JZIZxCX1WMfcEP9K"
            b"Qca/LJ6St9fWOB/TNiaxsEdWiWCURSEoIR1QxB17xhgk24LM9IK/ImS2tjHn+fscbiceIe56Y+ON31Vr/95kQ13XMTt4iNSnew+Q"
            b"NrJ/w5UDnDjAfMsdhmwnuwMD5SDfpr2WNQDqbiJQVdJHt6e2ut++0TH0uZDIROHiwbYcGn8wfNlvTNT5M15PY2sRrf3v1llKsrIF"
            b"oJKh09uoqsooawnR32mLnsRXQXmYQlKhIneGjrlfsGONm2crOqQTmFye0PJRrNm6e5qCaTPPpM1As0rhrU58/fsl/0/EK/+C9tkc"
            b"sfdV9bP3GIkApd5rQwWlmsfEHogQMYcrjjA21AmDlPbKJm00bIj0LBblViKJEYS4oXuHqMJbHUIvtjTG3NxjR/cpJYSLE002dxKd"
            b"NajxARLdk+PUiUvSgBy3psEhhrrElTYyA7zhkNf73dC6KVPqcnRFjquSJptsB0ubKf2Ux1sS7exUSjcqVi+UtNeL2Curqn21hkIs"
            b"cVAjBzFzMS2RpR+0dfPV6SmUEy7hW5vmneTmxQ+k9gZp+u5Q9d3hgu3oHNRNZozmYb/9aAUC1aNfT4F+ohvTJP+muH8nbfpyN4Yy"
            b"rLDAEktquY3XpkaYGoIcSW0ZBzKlEQyeWZxe9oYCeKNlLEA8IHiH+vA9Jszd0vxC2j4i/vYkOEL8pVBAm8iv2v2FumBA5S50bGrv"
            b"23Q2Ull6LP4UrajNZIMcWJrmFfC6TeWDbdQiWrNpRjZZebJdjoasJ7bI12x189ZNm22JMw1uVVRj7Xa8WoWPt1bLHYEohmizh4W9"
            b"034J8MQ66OaSg5ZZbboYtgt0Xz7QkYVtjlxuVhxRMt6qOfhyviW6uY08o458u+Ff9/vrdv/IaxCfSAJtzf7Nu6p+Nmv2Vz4mh2Y/"
            b"VATJQ6t/W2f8Y6Sf6u4C9dICfpAONLrBCE8NklyDA5rA55YEx3C8hDkBxanCoDRjCFmFsOIcMx4CByuIdlx/2khfbiNd6Q5tDrO8"
            b"U4cfvef1VtLRYpa0cik3G11n0nZ4Zip1pm6H14WkemwxO/PYOrGYrd9q8DkOLWZXuNEmMSrPfWafax/9D5rNfvpET/aJNAbmYrkB"
            b"ELM308EbH2JyRkA509b6eMEpKmJG5+A0xdZrLXnQQWPNRXjGN5CdjHHvqYN8UeXjxdjHSXOdDvEys1b5MWH1N4EtsckLtoXduVTw"
            b"zLlK/sS1UIwgmuX4F/b8Pjhm6zdyRD5yWDN2hXxUG3/qXHATlj9lNbZP3CQ3G8/Aib5U7xl4EyiJ77oFjjjwC8ixekaqqHHD5lyf"
            b"adML6UwCoXvU+wr+fl2RN4Al3+UUGLOq9XE5RxXF4DHDsTQ2mlFngwiB4ADEWcJp4NaDBBVv+ZwJhUOAIK1GP+zG+tZG/TXhyct6"
            b"yGPr14vZ8ZGp6530OAQ6jm4J27CVn0H24WRIkHdWcu7Re5VdhZtPetmTpDsEQJ4k3RtGrW8Bp5/nXOjy7OOcO3/lZs5tfFt/T0u8"
            b"ZNQHafdFKsfvS7tOIq8Zw9w6yokjsWT1FBgK1LEEXEdeBueBBWNV4mAKSbEUnhIXNBXPdCMKKvF10PUKk3imjUx+AtReUXvGc8Pn"
            b"4ecTVPmhuSEP2s2HtMnnmfOELTQZBp4QeybJdoROaURFOxm90oZGfIWCl1yLhyPWhEpJoVQWtOHOGhon8xy3pP9ef3R+gz6qkD7n"
            b"sXoTmXgCQmGnIBR2Cn7J+65A7O0zVmNTGgxi+f2CQqwAiZWA/QY//7sw85Pi+TGc/E0tCUCeCY2RxciBARDGKEy80MRjLA03VHDh"
            b"kAtKmERtpspZ56j0xgYn/P+acN89fbwT6s10gMhLZsfxUWvSkS9JSn5QGxAt2V92n5EcfejlBPINCjNA8nguhxtQDku3DQaCdHp8"
            b"w3eQS2jq5VbAGbxRkHpyfgn8HSJ+MzPUojoCVZkMm4hfRohXjeChiN+4w1vDBXf9qCXBfkT8XkuXB65iIeC1QRIHioJBCfLHuA4q"
            b"uCC14jRYYjWRijllfMyyAXMVvPDEEPMnsupzqL3xeOsnlFVzIkCt4wer5nTHLSZTuC5NnsIdjhDEGHlJlrWXIdksto97zZmu3KUo"
            b"4ksfa6HVrOjwg6aeQIt0yQJbJJulCak4PC0BPu08x39thHYPez0F3eFlGra+ApdAd3MlVXydBt/A9j7J9MXJ1GApUaxMndcqaC2p"
            b"TrAGpaiyJABoksg18XI3XhGmtU8yJdqKmFexE/ajiHprrDanQ6LL9EV0k5eILhAtD+oiY/oi6uiFqXKsAN7l77CLUisJkW60QrG0"
            b"Bkh3cCmknAfmGOgGVMTVvePA1k+BhUhJV5WThcYJA1uSFJMxWZSL5mLqmru86Fs/OVG7Q1J/YDcy0BfZHreDt+pxO14jwzHZrLXb"
            b"P/ooj7yOqgh5wS+0c046ClzGpX/MvlowH3Rc05n4l68EeOytdpQZ7xm23AfjJQiOP1SWD5XlQ2X5PioLvUVJn1FZ2LSqZtNZHJtW"
            b"1Y1C31Yesy29sy29jx/tSn4fKssPUVnAYZQyu3NE2JjWjbUQLHCgUsXiO5lVaamkBCyA08C4JiBwGviRIJ3+JmfAlxTbJ5V2W27+"
            b"R/+rLsOjv3ApL7eQ/Ro9KJvm7gXrQ5fA48p/qP7RHBxLAWpzI933B7X8xpUiu4pJSqO8cTmtfFHHEk2X1iIDm9OZ0P+FQenB5/S9"
            b"yLJ5UUv6QraJgiquzbLH3kND0Bg/fOhz+nsL328sb/dkxohxIjBvpGHEx7JWI8QZ+GCZB0GSFHxcp4Kj3BlHPFco3s1VDJGK/rmS"
            b"9jk9jOuadGNo2MUC+Q1VLWz6c5jxgc5Gd6TFjnTpjq79g012aOzfvA7l0MpNrOvmTnGoRMULG/Y26lY0o6XqaKWY8nHkLTBgiqc1"
            b"cLMVX3ItwbguaTerQ9w5E+Y4QqByJtw+frS2IjqBpvKR5o0YBgSjqrZ9o3z8JfbJkhbuVLQzdO7ZK6R6Ri6+UpWxw4r2cW0L1aNP"
            b"RfvtFS0zSDnqvYrrNI004YRSTby1XvtYwyIjhbAmCKeQxsY5pg0miGLrAAJVz0/kTrvE6hR3dn5jOLkl3MUIXEqzx+nbUUCzq1a3"
            b"yZc4ndDhQ7kty2yupoJvdF56G9+cdrNUnJic+fxBiYg/eNsyXgpz3o4Pc1wsQKtgvqXsXuI6ha016r2EKO8lxHGJyqYlKpuWqGxa"
            b"os4SYp8b+e9PdSPI7dvT1qFGZYYIL0ETQZn2KmDEnEZExdU1Zopzqy2hngVBJSUGc0A+rsohFqhUKK6/FUxLLyS1UUZ7nbQnWYY/"
            b"cnU0SuOag7ql3Ec/cIy4Kvx51fxZbq1VVFs/L9P2gfpnjqTxy30o1ZknVCmQSsbakdNAGrptFwyUQvNe0nurUp6JtZbrzfwOTYKj"
            b"6mfeR4polt931TVvaSis6jqV5VOFqapps+3j+xCAjJ7awVOdzs4vmT09cnY6nzE9VMR8HnZ6gJoyw3S8Skj8D1uV5vcWgg/K2RCY"
            b"U9h7QSQJROrg46uSMydiNUaCR85r86/nuCt/faJfdDf6vGoS9pYUBksCQxJqdDyehOe4lHGnGQ/1SpI54wkmG+XJTVK+TUMiZ/tU"
            b"YWGoGAx4OGIvUfEH29PkaBL/OJXmfUj0bPMxd1ZuVXd8Majb3ZK2Wq0gSjf2f/WE/bdik2psU/PaUF2m0wyuzJY2jYC16OO/fwq/"
            b"ZEJcZ8LL7chXpkGLg5UyxEIvcPCB2cB4zIiEYZPEWryMuTAA2LgAjdeX4ToQQZCKl1pQmn6vissD4tR09XoCM0Vn4i0DG9KeltTz"
            b"eg7SLtCC6gtvRDYaAdAexqPWZHdIvESRth9G22Zal6FScF6H4z370S68rfpEht3Gk6rxpoOtzhP9UPxFFX5uvFYQQL3zfgBDoYQA"
            b"EnKNWyb5rOVh5Yh6fVtA/m9b5T7T3BujmM5e2Vt4bYPvG5t5rIZ7oh7uCVfgng15tFxoG3k0fks4f5uDla58Q+7jlFgagpTEBJ8E"
            b"c6VzGBzhceVLpGTGM8OdL6teTIwwDmKRw2JBOIdybrC9AVH/CZr+L9KtostSt8xoxOpDxg4WESR7xMO6OlzYQrjj5afLZeNKDqPS"
            b"6ylqoZeLvQYcBK8tQbGIle8f2+4NlJIeL3285SwXbanDNCNFCVRJ/i00oWNqvJrnBFqx+ELwsWJVt98SSYXky+e9bSPHn5YqwU3m"
            b"u49/v5/7RlCdGW/oDPgzW0uzbfV7GHM07M3fA3av+fJbefcw/z2Eta/5DFUtPFxlxFlerJt9Vf5zXiBhjCfSB80YBK2T+KtzSHks"
            b"nKQ8ln+cIUQMgMeYWg8EvJcWA3vtFKIpyfh5SfYC5s8QHH1xED0ozYb8nK7Sg/bUbo47DgPtjkF1CVPZL4PPMZXksPs8O0i7prLM"
            b"qunBhWH6XmmLBHpszYhrOVlyVF/NZsQJxbgwh+hKlB9CPnNUOZeWA3Sz+rsJw5lJMp2JNc0w7A/Q7U88+v0VYWP3PiEAtXbvlyrF"
            b"NxGAUh9QKo+Ettw7gR2XzGOQRDqJHHfOxlswBabj6pmawKgNWMcc6xF2NrwFwHNlgktfugz+opKTmOqhrtntmNouL4jHJsMDiaYz"
            b"1VE0WPo/tijrxzhVx7KJY7ioMnG2U5C2t4ZhRbpIkaQotC/1R6CX/PriBIwUaY2M10n+0fqsBFdiTrzZ4KGW01uX2M+gZJ5ZYs8w"
            b"NDB41E6Zf32avYaV6dRJ35pmTyEyHFujCeEYB6O8BMI5FZrHFAoKC2V5wADKBBJiBmaaascxlyLo4Hzgb3Gx6UtU8ahKfdgP+0oV"
            b"egVY+YCVAgeeSXYPb0ciG48SnqhHp1yXbdkNLycOvaLILZ8CFpizmgyv1r59J4W3nh9lihzgRmS7GLrtcnD2FuAI5L4Zb30pWmJ8"
            b"OqISzyjB9HFRXMLKybfeAner4qe9yaCRZ2qVpMd+w3IN3KWbGuey1WSYN5bDvHEmg4EVzVhr+viw6DtBs7d/rJB+aDPwAyX26D4g"
            b"eBCOCQTMSpu8K2VMEp4TngCRRGnJpWWQSKJGSewVoVqaAMnOks7vAz9A/kGVEvF4ID9HHM1GUHTVxtvfddR9KN/09hbbnCQGyI1Q"
            b"w9a4xdF3JgC4KiuTs65IadLSYypOklJJ5GRBuG/jL9JN/1kOm/J9Bsz2L/N9+G1hxCPfp+zjPd5fM1nTZe7eCZauiXFHUNaP5SL2"
            b"j7dXOrERfvLwWwg/W05NVIdF5H2ZsYtdZ+Y5wk/+loYTdvRyMCV3zlmNwAMNQbF4Bw1EKEmMZRaDZdRyqnWwVPnEXowrqoDAMBH/"
            b"UgCB+NiiX0dHDlJQun3S543RL/mST4U9etJLVpuUyxxwO7yxgfmwiTGQ4S+CIakCpbCYfImlFkQHPhArMCImUd+gUCuQqQnPBuqL"
            b"Wsp2M9nMvNTIRT0fRgnHnOAF01mfL1psd3vp0Xw5yi9ao+cv9XpKlY2FYmunOEu2JVnKKnHK6hVZJVtZQUHliX7TXqMeXVI+5ucv"
            b"bSsITLTyHHMrqWNIGQHWJ3tcAsIh5RDxVDkpnUHWJSo5sgon93MqRaw3v2nmD6+Xh36av7mWfjWNErpxf8qRq0jmzkTs/MRJTmfL"
            b"WVQaHL0TVM7124S8YRG2FW4K2lMZrSZsx+jrEvuzaX5/Z6Brhl7cJLeiF9rm0J5KcyQVaytGjc4/fUA5JN6iYL2oMOoAowqeZO/c"
            b"9AfENcCzeYwXEeb1af2kLJkbn/B9O17bDDZPVnznsg+5/vHv0vmvmPJ/j3T+srAewTsfS+S/cMQvSNBOMOQRo5phDMgIg1ny6PPU"
            b"amXBYOoEctQQzbgVwtFYkgpuuYp16DetnvcFMn60RoazZbI4s9XYVqTi9jJvkBPEKtQuHi+6oV1Mz/q7XdHLVwA8BtTscTNB6uDf"
            b"w0X6wP5o0mTtl7bpbUvRKnirOI9rL9wj5WcsST30kMq92RSXtKZl12kdmj0VvnwKrlxtBwd2MLTNp7HXlvfW6cVX787ASpaUWOVA"
            b"qHmMvCb58JrJyP+r9O3bZ2zNnrjOnrhC3Ldsx+5hFbCLGj1ubv6OtfxiDrUkOViWKrMC9O2r/FOEgFDCxHRrQWOQNLmiKh5rzphW"
            b"bQLRI2OtRw4IxjJwB05bBpxxZb2TGIc3JN1lWn5f9WMzFTl3FHlCUG5gxr0P6VtleUo2rwxYgJByH2QflOpylZVCMd4Y1qIWPXus"
            b"W9/lwpPi8Tz7Nx9eVb5liKaoOBL/dQeVX06IeKX2qEKc6OOKxzdILhq50PLH08SmkKydQjfepqrq5gJubbbIfMr8KR5MREZf/Zft"
            b"Q/LFc8e+unEGqRPnMlvC1WiphVgdyZdrgdt1AHrb6dqBeiQCWhWxX/fX+1mLkBFk4KJByLtSq5XCEQPcxgoWaxmYFl4Kz6RiHlmp"
            b"MHYiOCUVR5JwLTHVhggElklv6P+06fRjvMAFX+orptMfH+l+6nXdg/svmk5njMkdAeex5zSeek7jqec0PvGcHrUOttYCH6V1Xqfw"
            b"j+f0GwyehLUqGOGBOy95UpwFkJphgTQhVmhlRAjKMOOklxQRh8BJAdZq5GQgPwWefSh/9Kipet8GdSxhd1/g+VTirs3GpTDFCA2g"
            b"YuQZgC1eJlNNnhxG5pCME1uIBa0E3miLFJhc9DZDvKNEXXtnWJYbjLeg576iTuefo2rnv1pt+ZrH6vgwFls+UORZI9R7kIEZ1UpO"
            b"pZm2Kdg2xaphBmNpJrmhZR/5k6S4eub1JwWZvuiB+rbZlmReIqMcp4Q4kEKxuCLzOiilQJKUT71gIlBFnE+OT1oTS6X0wVKtgv1f"
            b"83lqM+PMOe6OgdJF56hX2uCRknmuuuBlXOUNk6n7Vk/FtmlA8ppuVUheYtctocVWu9Z2blleSQy6lPlfqVsXr9JbaIHcYOBNV5dX"
            b"0k0ZXVH1dXnVTUit3B2k2j2Vi5gKr3u7lQ5K5ZdXQQTmNnvQoFg/HlCv7UJIbqywTlqjMXBjUNAOMWSJSIp5RlrpmbTBe8ItjyWu"
            b"5oo5q4VTjNEQPiCCN4MIcBmjI9EUhoPAtmVbd0K72dpl+MBAjaWWSFkFB9pP54EaC79CJRmCFMRWqj9CHYgNdfAEquAZz7wjLfYM"
            b"czWvY2eOILVg3l6dVhnze3hb/xKYQIqkgYK0RVxQR+OCXDlFbaCec9DaGB6E4QTJmPySlbP0yFgKloCkgvEHaY+OE598lWbKvYL0"
            b"ZnP1BtV1FSmRM5mSYXtgWkMuFWT9WYwr5WENfL9hWtkur+cg6nKzC68qzjq4qTc7i761uSDE6hUqK92XgVNoOkNV4APVe+zkLj6i"
            b"CK+dX8or5hWtcQqjDdPblG0w3W5DshVm6Zf86URK9NfKVazIPWUWyMe2t1Zlpcwiq9GWrPRXKqE+aFX75sM2tqG1ztVZKnm+9be/"
            b"qsk61Wyp5mDLzfMVTdY6H6OlACVdPh6VrWTL6HjJ63TpBqx52mgpnRI0aGmckS7EZBy8QoEFKYhyREmnMQXhGCv5GznBEbKUmC9T"
            b"pu4xDhpwF3ssSnrRMK/HUvXuFDtT4eBOkWPSu5JF8L1y4RjpIY9tSYZsgrzHEsMwVWS/upcWSiX5PQV1XXb660FsR9uQvWo52ob0"
            b"lNkFlCVHbWhYTif95A1WbXZaePUlATwish2OKAcuyi6UctpoEM6cTOTSQ+HrR6Lqw2poEem4UyAjuBM4HBx7PoASuRFqb2qt3k3m"
            b"pVewj7nqZ49AY7yGPzRPrrrmHXRXT93ycJ3dLyX6XyLJ2kDMZjiIGxyHd3lKXebOSkeMw8YoFksWDjQIb4ymygSFGPLeByEpCx5J"
            b"yzyKLxOLfMxkjhmhMPq3vaovpMzrNLFLNLaLZtJ78iXrOGmF9g1vKsuKYv1K+g94r9uTOndPvd0JaOKBdv9KC+7hbJfQxOVcMp2M"
            b"rj2e7SbBN1PAkTdWiscUQ6X/tVPQhtuV6HSqMVVjXH8dA5vAHJK/tmfdq+8m8zNs8Fw1caFRULIm+eoZyymd1kRfSgbVeZuo2/S8"
            b"Jv2/lbT/kpu19Mo7DSHW4SxDHmLVgcECxwphajSgEK924QFbZbUnjiNCLfWYOOuN+CZc8BfxEHhqB8UGfdqjlIGaS2qfSFKN+7N9"
            b"87gXH4C1z0GfkTdLe1oYYEDaadfKQ2j3KkokXajCC+JtxRU8BOv1N6cBeg4fukDp9ZTHl+FlCds1x5u7UzwsSXf12KaB9Tb52HHH"
            b"l00lDdiUZcumkgYXDKKW+dmPY3VvsG17RG78pjIycYTLfbmqQSwPNXhqpQ5SUKIRUxaIjc+RYMZ5C8JxRRGDAHHx5mNGi+slhxTY"
            b"oJ3/J4Bc9xC1cxTXZYjWxbx00ah02rI+Urquwb3oIjtIMdlUB3k1e+uFZFPYRfjW3Dahq0e5rBra/zJ661ydgFQ5kDyUgjlHbz2n"
            b"TvDBb92uD5PplDNeIxQMist1JgGMRQTHlbtlQSpQEuukvimlozZZFnirBbPMBir1a/FbjanKnKVwPkg7GaGdennWB3QD97VUREtE"
            b"ZWPf7exgDLqsC+WpMWg523Y/aOkrLyQ2UQmcdIwuiUrUmmt3G9GWjZZzFyLw0B40nWeOiYkc7Wiyvfkie9ZZRlht2XYcmCPgSw4C"
            b"LBa5Ut5y0INKUAUaQRW5Tafqx3Jd726jqvqZPHGRuqD9R0nZ1++HTY3mUt8BjjpWg4Dj2tbwWBAGTQ21GFvCWcxZRAfjsPVeaOeo"
            b"wECCEpAQU0QLgoNzjNi/Z2p/0V/+Khn2ouTreNQ0LNruONuPIK6ymoF3NM8ctkyx6rYoXYf7m8DTE9jdvfcpVcuPPaiLX2WPZYJT"
            b"PGSFZDPZqQW3jwebosvpMYoVb7YcNxLzIRdkAWI13wGPyRE5rOw8/qkg3KtuHy+nEkafNrOn5CUwrZl89tkr+OhlUL8yKENh8Ag/"
            b"ePSxsv92K3vFdaw/LZXBGW1F4OAsiSlCeUIDISomFuUVd8L5NIQyyUqaaQ3IMaOxfq2JzL2kf57x36LIfcm2/oX0gBSR6mJZBYod"
            b"sNTUk5KuyTfeDfCKvdpcp45+98cSefvgD8oAXcG9/I3hozzCxTumXHuvNTFhF8A+3LJYcaellSLE+KTKydOiefDUGj9dBnfYsDM1"
            b"gpkVNJ6u/vG0a9pwYce02E7XgEP98B9LrD+UPreUaRg4qoX0xmAuiU0mBlTHBb/z2inhJNfUihiFPPcCIWyT9ZYAyik2X6qW2Q+A"
            b"Y5/Ki2yIjZ2kRnoxOx6Lw8taBRepXScVuhhgUEdKWUMV/weJfBuIdx/YKXttIH5I1sMGRRY5hE1zgI22WBYDKTiu+/HB13B/N9bV"
            b"vlDuH5Typssxi6d8eQMOgh0aI7RaAe63sxLamh7cbGYQpET8PO7MtOauBgVEy2uJAw7VmH7uacA2thiHCgLG6z3DAG5bPeTL/m7/"
            b"/h9jjz30PfgBXtmorjacZoiWxVZ7rqQEbJTyTspYRZPAKQWmQLEA1lsUQAgmpMSCMGuQ/bexWxeRVBfVxS+jvK5Boq4Iih2qV1q0"
            b"Dqqmy22I103c2H1E2DWoV0FtpUOIpTTHzS0YV+9x0NPN5o0ZtBXXj+2nsvE4mqVCjltK++eQW5QyBkBvpXdK6pTePis4gjqJyyaJ"
            b"r61o3jSmu1ehfnHd79aUHig5jmC6zZPfIiX+D2G5tOIqoKADtwHFMoNrHzx1LBY2JMjgQ3CGZEEGJpDSRCgkjBZaEBesEv8gx+IG"
            b"D2LmqoAuuSqgzv1gzXoMWgVEeeAP36BAlN3l+wwTat1vQ4DolcRSXGE1UI7VhiSo2Qno3C6y9rVAnW1D6kbD0JetiYwxawHPBe/g"
            b"DEci8TuJMeUIcjDDquI2l3INHfwx4gsl8FlWxYt8e2eUYjw1m8RTqEbtx7uSI/DW6a5/NxSTvOzl+6FQvKcM104qjIJwgWDpNeFA"
            b"vNRWMGa1lVwCeC6znQ8RIWDvnMHUC86liTcI/72Ckw+AcVNU3AuNgXt5yCmYdwWGPevyW8N5edVJFzt3/t7ksSUS4/VWC5NdTiUY"
            b"RC8xsdKZ+VFWrQ9Ob54jAWFYP0S8dUYPYwlcBF6QkFVwnm60uOMc0WA6bjui39F4lAupGCq9xtoJB/6rfRyhBm3U1W/zykiXt4Ft"
            b"dAo4fJ8a8r/g3djg0UpHgtxwYnxer/Gg0ahDLGHjBYOQl45ZwxzF4DiPZQUiCrSxiiLMFAgv/5+960xsXcfVG5ofBDuXw7r/JQyL"
            b"CkmRsuTYPklu7nuTI8ewLCk2BAJfYQAsWOts4E4bRbnSH5R5oJ/HfbTqDfTQdvyqMzo/bXT0I7a+rH6kCdHL7NS+uU/a8JzLNRym"
            b"a6hqUiyQObnywIZ9gSKfkKOBEcFmc8kjL06UkR8l+8SPV+W4GOp4kHLSgJAczAmPK5l8SDn6yWnhTUzGTLRxrWd3acY9Xc7lduYI"
            b"DzzEZPwc/MVUTOcNgOC3CTnoIHgALhS2BIKhRPr4CCswxCsZa06l4q99oEJz48GCIMQ5BJpJ5hX+0xk7jNGSglhKgAUxWwgHcrHy"
            b"ebMy2JuMxbK3enK8qGx6Kq+L5l4iShX8QAwsXZxFDExUbhbyUMvn54G/yarsvt7N2KuMnXiVsROvMnbiVTbSROAT1Zs/gbHLAmPJ"
            b"i0wZH1fYgjMXy0sCliflAkMUd5j5+IFLNafLbDLqXFqMYxZ0AMn495t5PXT8ZmduZvLRzGvlGbx05sVOkVvbzIt1BuFfn3n1e3zJ"
            b"zKs/m4tqBSejseFFvzrz6g/nbOa1XRX1aOZ103OccCmQukVEW6Zcy6SqfbSuvBv2BT9wMQZTrXVhT6uF/bpdzbsOS/tVwHFPwd2D"
            b"j8y73uNPfpx3VY5Vn3IuH/RBY8UpmRFGchvTrrA0SCmDZj5oobh2QRmQ3FDtLUiivfJaOGyVQ4hZLn8ew+Nxx/AqaeMiovUyp+Q6"
            b"CeQiE3gk97WKZQ3BaHfIJfs8CtioA/qQbFz3KA68mbzT1AJVGKMB25cd1XVyZEGtbQzhhvA7eJcUmI+IYUppNQ3cJmADSeN8McsL"
            b"OBWc738EvhGdu3tIiXuS3UFBUZLmj3egaqR23KkenGtBnt0P2On9YIZOq0roMbrhAHTgtH3wx/34OPfDUIrih1XYmJFMII7j+IBi"
            b"HozTjCKLscbxK8qppQJbHpBhgcVfGWsUEfodN4WDmMP58OoJqPILuHqHEdENJlsMXIC5LY/tAH9FW+JVBPcv4Ec48uw2dbinXGbo"
            b"De5oHcD8Av9wAjmGXBRtf+1dC2yFrrVXOF+DHP/d2Xz3J3MvwTzM+Hkw5fQ93Nt0a0c/zOJqHMT3H93VeIYZ/HjJ6xfgxxeGfZ9I"
            b"7crF/1DwylivrAhUxM+2NcFqzZSyQB3VgBUn2MbvIxEGxfKfCCY8MdJ889YzfoR5eOLO0bae8SHLTVrPudzsUGHD1vOx9TtpPcvr"
            b"dhH9PeCs9Tz0upi1ng9jzHnrGa62ntnD1vOaO88a0O9FPsC0DQ0nbWg4aUPDSRsa6rZHY2x5pPh9nzTalMG1zPrXMBCv7UQ7Km0s"
            b"bBXinDrBXKxovRMSu5gEwSYVHkd46kHLgJTUlHkjY7VruAFtzGtZzS8RNjtpfpzpbB11AgeL/34vAxOcoZ7WFWIgKtWxamW5j7y7"
            b"vHjdZ/Y76rZxQF/Qrkjx1he+zULp6aLBAxjVVLP+bVWuWnEW1sH/q75f9bRhvT/kmDVhrYGvm5fdHJaNyz02LerYFGLApiXiWCN8"
            b"3yqF4I/UEPugUtiekrxRwlglmBWYSUY8UGU1Dih+2+JzKHDuqQXvNaFgDeeCmoRENYZ9qiL7RobnFRxqbJCzgblmRjy11c0jo5vL"
            b"JZZcSyxAi2QYWp1jxgxbXOpGuSqMqaUT2rlQ5IiNSbx6lzdBR1QXVDioxgEiWZLFMLlO/kSNQe0PMJ5BClyFyHDth9NAxVYRsue1"
            b"xeBmmsMHQYVHOtsze3I2tSevbQ/O/G9+gNPNaQl2wdHmlSVYAM+oSolOqYBVWpJy5hS1FJzRPtAQD4EopwTHDjlHgkeMxCypCAro"
            b"vwd+OiSKIbBpkCgGEKD8/J/P4nfzWXxVr28mHzNnMc05USNWU7X8/LNZvJv4bCzqJChjKRKI8qTuSgNmPoAkscJT3CEJwlptkAaZ"
            b"msfGy1j5eS+N5eTHycO8XlrloprMFVbQLZEWvg+7cSOFMnVTfLmezZM6MeWAi/wLobRXcVmEunpNnBya0+NuZ3Z4FektcAoDt4ze"
            b"RWXT2IWWK54Dk+NWuw4/7rnElOP4JxIxt3zEaaYlVVq2vGZE8Zrg1DyYjevlybi+wU9VmwVitfCkilJBB+rfHXQroH/pOP6pwvwT"
            b"VRgriQPAKpa42nAMxCIf610pgARJpUEIe6UsxiCpi//jPn4tNEly4Vgq+gwdtQAw3wCcnaFmvyATc5jJFJ2BJb2zTKztfAjTs5vK"
            b"AB1E9ADYoeKLyvBWsrD+Rel4blObzpUwr8FxNQZajRJ7cQCMpqR/OK9Sh7ILDK+DdqbG0gBwijemhwucd5R1FcUKpVrPPdfTPck/"
            b"XaEcuta/S+V9z2OG38Q4jSvXNMvH1cwFN+aHc1vEsShj3QPY/bZW6wT2cJb9If7+A3uZB0orq6nSAH/6chaqdclIRquYxoAqxaVO"
            b"HzbDaVBBSWupkDH9JTpA4EIEgmz8H4SYH73BKvwkgZWLHq/t2njDlJOr8NUDyP2QNbZCT75V/mXgxLWdDVTiU/Ip/akR4PRMBusI"
            b"OK00YS4BTjeRRDUCnC7cgKKryzaJmuXEl2qzu5Y5MB8RcKBsba9UrzgBxabrWg4vFugKjV8Me9Xd8T3S+5UXPivD8kSLdjxvmj0z"
            b"nzc9O4lqt+hgq477k2H5tAyLg1jm2pj+HVGWGKalwnHphmP+J1YiFiSPRa9jRAjpKQft4kuwxMYLhNGvpB9c5gtcIwHcwfT/O1uL"
            b"+2yCcftm5l+2NB0wIbJzMBvDRBeLWpKuHHR98IUcMHzdTcuNcrqL1G7rUobX0nz0qhSb3yJ78rQl/8wON51GCX+Sh5A+R6+wmZgN"
            b"7+S0CD93TDuzlGg7Hb3vWd8H+aMafJxq4HBIKookWE255UoJTywnNhgrKeigndI6eecqYRkDRJwmcVs7zinz7HdTgy9yfi+qDV7n"
            b"Br9MxvG6wOHYjHL5IuGuJ1PcKAkVVWtiiez6GKSIPRIiVC9T2IQJlVM3paQ60oxcKe2Ortuy3nwk9M6SbCBoRskSColIk687W0Vn"
            b"2ho9PV2aM89xgO9ZVMgmkdJLk8SCm5CVDaWsnhnZ98LWNxmzAdZ0fA/c+sf0vcv0dVJZJgKTlBCmjbSOgzRU2OAZM854ZMAobZH1"
            b"AnOjrUcOmBGCW408/cfyX19E8V+T79pK71Na01zkqxtXXsbxj9QOZlKC/QpiQuZaxbvI3pc4nm9+PmdeqdYSd06EKjII2VC3qjmn"
            b"XZ1SatLWf3dMIyghImM/MEfbvoeVbD7YEpcsU1UpZebttRKU1gC1kNdtusDN4ndm47toe1Vg293Gd9zPfqjnxfvNUUnc6339ANrA"
            b"tZK2jPPIDTrBm0S9nDICEJfBY6k8ti4wEBTFXKu0VCgEKxFyXiAd61tvDTbWao9ioFdkjmv7ya2Mq/2EG42CW8KJV/sUF43V7/Uo"
            b"cJXU0Sq+IHb5BTLQmUG5Raz2nC2alkOvNpkPJF8yPqywHxhddHZtPWTv3ImN3zZjG+BM0nHn6FhUC96ovA/PuFydEvxkF0MSKTmS"
            b"N1EdvAZkVA8Wo5/avIfWAJCZY1C+M8odA9I+WtR0t8R9gHJUm5XNT6OiU35f3wb+uhuf725oEIThrGlGaHCSUyVS69oYFrLaOGgv"
            b"pETAOdFchSCCYVmfl1rFfobKOJzeWB7dVdQdlfEDja1h0bba2DFMshUgssfCwIWnky6HG6o46pp0ORyCvyxdftxlPN0UGu+6e5k7"
            b"ERC6qlmJcqtE4Ke5a1TeaEjM1MVhqi4OU3VxmKqLN5aYFbd2Uxf/Qe6Yz0ObXwrqcA5jqiVYoXHgwlLnvcWcEEsEdlRTFeKmFZQr"
            b"7gNTRBuvKKPaKMlB/Zdr39fP0i6Ws/dmaFenkm/0p9/t6ZMvZaNjMB2VLXO8NPtbmyEbZqKelsHRLJiUfH5n+keXI/sBw7z8iXtF"
            b"ZoYuG9fbNY4ZX3qmtc0Z0k9q9snA6pjTevOv5v18zRvAKJ4GHmCTDTLHQltsWFwZChE8yJgYmTDcSBRTqI6JTVvkY4wkXBn0HzOO"
            b"v5eFr2tOPrajn6iT9T7KOeKlhvQnJsH3MRWHHsQjDbGD+3COBMYR6pklvOOUlKBn3ePvlcIztvKaYhujnCp5rnxn2vGg5SbrUG/D"
            b"1l6AreUwK48r8/jfmVz/rXm8J5QqIgSzILS30kpMNNZaeyOci4fjmUaOMC2o5AFxGkCYgBGL+RR9kR34YOX/JX2aO+v/51Qary5x"
            b"Rxl62E4++hKPBnSnrd9J+u09hXY5ct4EislxHBu4atTRuLoCyGcj05+KrM5Blc7usu8h7DnFxyqUQrmwee/roY9fV6KLJjqB7XVy"
            b"1A4vEfnv9mS2vWsyUUTHap0uekFSYpHzIniV8KoesZyYM5VvRURsD2oqyY5B3qrXZhP/D3e//ZG6Ol+04nkbKsITroNERjmpQpAy"
            b"Fq48fnkwcjG1IokscgFLgZiVVMU0TASKHz+ksOPCoRcLgvWJ8xzh8KJ+RJe7JjJgB/XvoZ4Yf663cS1d8VVhB2grCS4n7g05MGsf"
            b"4t4yd1IQfrUlMsSLpBPLR0ARZXUZPUM45Lila7EjHPjbMQ6plXAXP1bPxWgF6qUVkLetUfFuFNE8gJXqXDOdawTZQQGx2uzauLQS"
            b"Ufx1SrP/Tk92zZmCY+2wAhvi2t3Fj54HxYRjnhETjMTWOqes0zoRmb2KR0YkFQZTQAjhZ5Bk7ATKe4rjfTWn7yokN6e+dL7NrIq3"
            b"2gb5eZn+6dp+BJ8k9yGbb4b46g6soiTOs8kEFiZGcNwcQ7Ki4g6dFRWPrd93iktnQqSCkZf6oT+eLs5yDyEIDejNXWsBFdAZqSAR"
            b"bELhK1H5fleD1Ba0Qm+zlg64xO3w3JsEOo6B3hOdnUvLllwqq7wqLwjLziEJsKMKms1jU7VLuBsC4dvQ5wpN7oFJ5Byve+BHn9Hk"
            b"3oYn85oGbbCXCUNGY4ZVWFJv4gfFhsBM8CYmBmUsMO5ECJQw7RTxFAssrCXf0IThhV3Tq6YG17WFjp0AvGWBgQTlZbOER6gxceBB"
            b"zCBaA6Hwc4jWiJGeGw4cV0Xrg1N9DDI7eDKkwHLFrzlU5ItYgjnGGFZtz+UQawRZ99ISns6H4zUlL7gHvAtqjkaPOTxfty+5NOQZ"
            b"nLxpoE7/V6ffa0I/tNwLKqpG8xyvaRy86Q4PSW98uDmUBPoghOzPs2Hco7AQCHVGMhk8wQlczJGOdTYNxpsgvA7SYGm9oSbeHcAg"
            b"JFFQWiCuOMM/STLjoijFC2UwbuhbvEHP4z3aF09IUzwhgDFzXNu0KMjUcU3gDuyA2p4zOYpmZASHgsqjbTuP4yqjhG4ACVB9cdn5"
            b"9UAuQp/VxcgftXu3AKjaHDXRY+wxIaeSRvJ/Kze65knjg6XmCpc4sYA/RG7VvnyMh/vTx3hT6g/BcGGJQ5aR5E4hgSvJiQEdqKBc"
            b"C8ekFMQiQ7FTiitqZFwDYB8QQf5XAusuDrf+Gf7uKlTuJp/kGbbHrxDJuOILN2aWZF+4TypkvErHeSyQAVOBjDOF5953rQXXtYC7"
            b"npHdw/H+4HQfh9MFThxxOMhY21OwgVEZjKRWIBkwSKAiPvbCBQUy1v7eGm0wYQwJJEDLD9K2Py4wPe7F9/rSl+nau2+InDmH7M4j"
            b"ZOw9ko81nQmuSDPL+bdeRKLcJlq5fhj5xhXJfpFVLPhALfk5J5AUki6vQFVOXdZH4iDwKdAOgVFTRehWEjS/gSrtci4fyzazJTD+"
            b"FMsMMwfTI3AmR0CHxbur73wzW0PF6D5uH+2THmf40TNTsPODrR/gQVJl4EWe+QKf+4I3yZv670FYnDjcFgIFRBHmUlspQFDFgwos"
            b"BIcJSB1/ayUjwSDDHdXYKOVB62+nTIS/pMB8AMR9VXOo1zC+rhR0aQJ7rtIzcATJUWUaWIFFykpUjMKX/AkEVV4lR22hK2JKZR95"
            b"lYxWY7pKAINMHEm3NtRxbZHer+wM4ppF7Ci9fafN+6eY3GLZoHlyWiYfsXk3+yX5Q3EHZpLnnlBj8x4LFS1oPFlD856TKhoPQD8t"
            b"VvT17sgbhp9vg+XFctcqQpGhFgeNg3PESm+coUEL45E1DKvg46+NMk4B1aCodgxZpwwj30oPX8xzMjzKyS/poI8zbpeYHmu8LSL6"
            b"3bRysf2gqCshes37S7eL63eBLAMXD67uCK+Xs4tMsA1EC+54hDjpa90cl+8fclFrqw6ieWFXeRcPUih3m+qmuqxCq0NKMenPiHG5"
            b"bpXaxUiyP4ctdzPKSjujP6hufZGPvQQ/m6Y5vc1XueswNXtmzn7Zq+KRP9W8m/FzW9dfSs/Pc7xJ3iZjvnfQylpCMeeUBK2x9sEJ"
            b"LWXwMjhtmbYK41ggM0IlUYxI4SzF0ioU0zaG/56HX+ciDyPo8bhx0GlJlsbBIwO/YxsAVamolZ5LQUyiiS39JUW7S0rSY5bOmTP9"
            b"sRWcjrNY0z/y4+ObH1/lI/86S2Yq72nhj+d67ATgx06849kJxI9tTk4tz+TFtqX/Leu+4LiKS34pKEJWGoWA+ECpiYVp/OkNcCet"
            b"BI+cDRCDreZe6ljCaqPZXDvzH2Lx7rjuPcXKGw7RLsL2zvz20Mhv7yoV4ybbep+IIbaLEs3CJ4Tqo5ffXCNjBL/DsCEmqnqerK7V"
            b"O/Ol7ynw5RVppEWX3Iqr5cp4prdAM+KJEihuUht6b0aijhenBHPGEDxmq5Sw+Of6EvTuFuoCpjn4nHYyxlacSyfDuEA9qFUcCdUv"
            b"b+z+Qe6em7oxiNknUEaw1VLFJG8CMS5g7zizQQbhjZDaSYy9D9zYuBHvESRgji1xiv+xsf/Y2H9s7D1VJ2jFPSPVGR1bTunY8pSO"
            b"LU/o2HIrk3vrEE77zY2OLf/o2K/v+8Z0E3wQFBuRSmmFrePeIiawkFgqAUSnAyKeY+o1ElJoHMtvQTWxilh4h0d2o3+5DpmOs7Vz"
            b"cMMDZMPDmdmwMOd7IiQMH2ype0/qHLSQ3UrSOozwv8rCGft7j5w9Bi2Q4uqRxhL4aArdUh1TRPwBe1E+CqTLmaaftGGMj8/4mrt4"
            b"ft+yyyQIxHYPWlnvvj7gHJZ3yGGL55WcJ+twJjlOcPlFgENOu/dkMI60E15LVvAqZW4PWAEx89qTiVdurGOBTrZzCJvNSvuidDIq"
            b"9YsXNjA+YVX9HsTD+5IvBc2l80QwA5CUiSmgmIcUZ5xzh50ggSKaJDIs9hD/XwblQGjLQM0RDz+C132NZH0JenCJY32Rr33gksyI"
            b"JKtg5QrE3QEEA3YELxWppI+NUfP+UuRlvvbV+WMrIVLuRGokRM9ldZvje5+az1ndi6neh1jdd5TdlsZDNwN7RP+gFQqYdvQPONbG"
            b"Q8ePgUlIRfSoWYJ/PO4X4shYcrkmSri4SMPWcilUUsmItY5WjCvppSPSMiedlTgw7yAITiynPChKrftNepg3aNNXFCwfeIG8yasE"
            b"LW3eZU/r4S2VX9NKTV+ouHBZq9SpHma6GikuyQ5v7GexxQ7IKGc+HuIeQ/yGg0enkAwl7tmmAL41Q8NV6Yi7MnIsiMmm6DC2tA1g"
            b"k8rEldM0nhSlx82tg8t+jmL8z5HEZMAIWKWt5AoT5QImFnkrJKNcCGMDFuAo9iquxZimmHnNmYe46tHeEaCfq0efpsnBAzgBm4zX"
            b"LxdZaxpdlqejrueVxmxPfKgvxZktXZc7Bw0LWA8cDUs5tNS/cjsFfgI6ILl/kDu4bI89MYhiWxUu1rQrqy5RfxVaP479LfYym42a"
            b"xGvnZWNBbGfd5u32hdnKI78GyAb4qI5PjOz8xBK85eS81HyfbuZMhIj9b9WM32dssOXemdAFO5GHO2AZJqiGLi8/bNI+LmnF1bQM"
            b"fVpmN6AO0y7tvKQtf9tDSVtBIN5V0gpLYyFrOEeKCBRzsXdB6FjGMAM8WGE0NUQoCU5zCDz+H3iKhOReiLlo5teoEfR75OHr5OFx"
            b"wr6gbHG/il4v63jyNYBbPEzLTQ3M+o7pUd0nj7fIerRsL5rHmTYHp/1XJbGo4rfO7MESD0PhZVBFKO2Bsov96FGVgi43QQZE4fGr"
            b"WvZcjss3T4Zpc9HYieA8X6LjT4Kbe8GG4z32xUswybgMTrtrwebFeQnfMBY37wQ34RW4FiVqHsBJd/g5Y5Gx7ud4swJZdJs1SOPb"
            b"3B5m6OD7tOa5sMWj28ZnABYSoaAZ0tpqSAKi2gVPqMDWKGEFx7GCj3mVBpDcYW2EYTRYhKgyyM4BFr+DXXdVZPQaqeJUj+gV1BFy"
            b"gTxClv7HAKWWSvZ0txP/qw8PDw0Ch7PEgSxppofkeWL8TJXrI/bGNT6yThJiegNBbyi5lcDRhku0Uwu3BvZ64FO+YJ7m5Rsa4e0R"
            b"4SN+OR11ifxQV3oGaDvzsqYVqa7ellOC3RhpsbtU753oqlP9R697/aRPBmQpYvFDGRBiRIUQCFgcvFc6FuxOJuYGDUQx8NZJGT9N"
            b"3DPNJcFBso8W8F/EvcEZ7o2NkszlNPignTEYS/X4i77I/KrE3XjxMcRfDBSAZvgLdBhdTvAXvfLGAH+B9vK9D887xJuS9Ao7HoQN"
            b"FKTRyluhg+GezML4gnWWgW0XPgfIpzsnZY13h9U8g1rACdTivJg+80KVDxxPd6jFwVLvi8n3EwXyG7onb0u+WMWPpPOEWS5i9cuC"
            b"xggJ57B0BjkfTMDgffAoANcWBaZETMmYU0+EhfDageBLUMUnc8ImHWTllyOzt0+p6XNSt0j6/IPz8xjRTtXhQgbthn5pH7mjK/BZ"
            b"yskfof9Rua29d/xDfX5yYQAjxWtRn651kJ/OqftRolViScbxjoT/V33/6lvs2ujJMSvdbQ18HeWN3tbKaQgRFxRxZio6uNoDPu55"
            b"ulWaxD8SnftBDO6ak6znnEtMqMKCWCuw04EypTX1ARwXoC3nYC3zlkvPCfbJmY74uFz3Uv1GieG+dTquGlDODHgVY6miJsCvfyNd"
            b"PCW+DU09ThrCUyngs7brVAyY0k0MeNBDHcsB5xMoL+BU7BlUHsrp5fo/oOjBUMdtaHs6DE1B8QMggU2OZCwSHZ8sh/+s6jDB9+AT"
            b"Y0UFNtWUZFN9BjZVqDyM5vohXbW5rvr7rasY3z/J4fd0ZnGI+R+YMFKTQHWsUmOGJy7EUtVyGrOIZzxoAwFhjJ0w2CCuvZLMaSuU"
            b"fQcF460qkpdMmi+TpQe5e1d1eXwXOqxgJ64kY073NaLZTUcSXLHtUDlOXlqXaFQoHxlxM13fITYNozla4wSetgz2ukNfdPAVyiLy"
            b"agdai1qk+HASOXaFtgk46BrPnE9K9KKkzxgWpYxTc85hier0hu/SPCSRkiN5j+exeDZt/VpegTDw8oivt4vmITvxkGJZ27m8IP+i"
            b"e8g2l2m2OU+z3WyK/a8MOpcHm7FUB+8ov6/vHd+eCdLM5mYk6WWxeIEk/VaOyOU7BZEkLgCUkwnaLJIBlVIMGxk0sVjFdCkDdRIZ"
            b"xrQISniv4gqDak5YXGwE+3wjYxmxf+EecX6D+DJyeJjy2zlcEWBgl5xV4eCsOvVC3VTQ5FP3m6tc5dlyYghgwIWMkVBzDbdueP5l"
            b"j/m+R0gjd7yf28FdNV0FopBqqnW6aqoNL0gOL+8BHFO2ylbsMJnRweXIdCZfI+bFD0uSXbqvgckbEUy+F+B87wO3jwoOIyfMlRZd"
            b"P1qVhKBWEvqa7eqann9ZUv53qXdNt4ojJIELkiSGuTdegKSca4SR8MHb+JF0lkGQUgQKgSAALUFy5pVy/g8ycQs3/Vhj9zKu4jJq"
            b"4WSguMIKOgphmtihFQW4YwBRjxTZIMi1fRSZ7XhCoNuNnZrgFMLy5+OBEnJ+HuddY44aaaEDklmqJerogI0ObZ2j/fVdB6dbwzu5"
            b"DO9oNbur8Q2zZ6BWMa62Z2M7uefXZnOsTHH09ftDTLxuaJcYfB4nJJoUhDmOXFJ8V+CF8UQF58EQy5hUjguLqJBgg8Eifq6ssAZ/"
            b"K0HiWUJ+YTZe+jhbWtrfc1TlogoG3IvniBQmN9WELbZBdVxurcsiqyAZWVPKBvYa7HBQ2Q/vFnLL7+ios3YIZqREEkW2o1BrI0L0"
            b"VWqMwemffAXEPIyV9jvipAT2Gu/5GYK38d874WRnsN5ZgjxHL4xTZ4NUmOAafohzXUl5I6HLd4r89sK+jPIgtPBCOKsR904GIEE4"
            b"5FH8+Jh4A0/NYCU1505imyTXQ2CIKSotGE6+XY1Z1wrT+eE0q8nX1pi99EDb22W1luO8pwort2A78uvJ79BwXWZVyxt3exzJrNWm"
            b"o93pXB6YTqgj5MjvPvZ7ySazw9bZ78FRtRxvwzchy51jduxDvkn/XvIB/XlJ9Jfl1QXE8v2WyDrBdbO1fQRLG7WGlHF6JGHwBjjW"
            b"PUvrJ9f9lp7sOMuuLIvScBg8+EgpurUTMN2/1OVbLPYv8tdL0TLeW0vR/Oce5mX01lKUSuSdgqAx9SRWmxqcDiklK4M8VQY8iXlc"
            b"AyPCOssl5hRQjBaCMTOX6fm2nqAXSXav1Hi4bvZ5sWF6NkQ7cNje7Quaj6F0WDFeD2Y7lGP/tux58YbuvTRnHMESu0BCKnfofeR2"
            b"3R36xLWzBBZKHaWV8hFU3d+jZDHGyws4FZzvf7tNlK1vMZe4Z31B8f3a+egPfaZhPPOHPtMw7qSI6+7uyIDuEHmnyv6zB33TDI7q"
            b"WLlIK7RSOmBvA0WMC2Wsl2CcMthArNQVFjHzY6u1B6xdrOZt/MRaK75YsJ9U668v1ed1+rT1UNJx866T/gOp3mLUeVgrSLLmIX7a"
            b"LT72KAfd4v7QswEbX/Urdjjhsre2iZDDMNsOr5IsHmrqX9SuY6skBsGiWUc8w1TPO8k30bXpcLNEvoc6nlrBTRTX1qidp4YrUXio"
            b"ROGhsZwbJcM+g/7MkveDhe2avxzzJpHRKGArhcM2pjBsDeKKYuDcMcEDKBcPI5a0WuB481ZSQ0x6Gnv2Bo2ficDPV1QlziUlznV2"
            b"SOeQ+z+8qXUvy1nZkjdWi7HcNWBbLJmLhF3S7mk7p+suZ0piKSydp5JQh88qshSWClIE295F9QJyn/SW9pQDKJKsMhMmvYC9zNhY"
            b"JMVODusC0uvWBHaTFJbqRf4SkwpoSrfWiuJL0IDJur2r874TBeyWhM4HhXK2jGYUktwHFWRcciudmYVAbcximkmZ5kWce8Yxch4b"
            b"I8DGBa40WADxyD0zKVqXkmeyZc95TXzYJe11S/jLzhSVddDUm+0wzZKHfDv39Dme5plu+3gtPZjfr74QE4UesUj8shW9yttlNFyY"
            b"958s1dOOF9Xc1V94dxeG0SuOHsOloB43YL5uOcGFEPiWxTsp2XHNnd3DxUuiEierH8HCEYQdZVU/etC5bTqs1bK8QsPCjpJtVt6l"
            b"MfOz195fMqJ4j8MlY4g5rTwOANYrhERcQDsBQXInmGScOsFNcJZqphhxBAKzglHkDUYs+B+Txc+dK56yljtkVHzwYbtscXEl9V64"
            b"axzMefFmdrV/2vbUuXoaV/oGux8nvuFE1FMpDr6/PXmhO5RiCZQxsWjY2hUHjw9UBmxA0Ho5FogVHr4ghxV/YUaQPLCguwNK16QE"
            b"NriscZv1gMp6yuaNxHVgTOW3JNZ5oRxsubh6JIvvT52Lq0frzGxL+93DnSqx3gXqR7KuoWWd1rtnKmZD9+Az3pzvdH07Nbe44O72"
            b"roQOiHAfK3CKrIIE1KIiIWqJcZwKS0jwOEkNO+K1QwIhquOWlUkZB89pbWeNhu0PNrEtnnkWnybj80w8mMdX+ulj5vL+9MQ1c9wf"
            b"eMrhjRcGriRb2SsnsAe5wCMUrzLleAZEc5Bap1+LDtdYgXG/n7S94a4iFhuiS60F9NT2E+e3BoSW0lyudS06kBUKrjVF1ioMd1Mi"
            b"wa8yHx73FNgUcMWmPu8j7u5dRu+nU1tTkK6YqwdS6A8T2Et7CYwGTWKdFBRzoCiTYJnUNn54JBaEW2qM0UQbk2pVK+KqSRBvqdMy"
            b"JjH+3+4lXFjXX+wlpEZqqlUlZrwzyDhUjzkmHZkkkh5KvApAsqU5uWggANBt923+6mowvoTGn6IyddiUBvr4VqxxINWYSrS8LwC2"
            b"Co2vgIAV/9IKO5ZAAFHQpjs7toSjvkpuhSorcGwnVJnXBRQQPSh1tXtM+8phlOO9MB6+eTnIHPe8UcWN2hNXQgn1tpw6WMipxI2c"
            b"CubU9j5To5+9oKye/esTvLasFKC5IRpwkIZYjjizxov4OWYYExoM0xpM/KgIDp7wNIC3RGrFRSw77Z8U+plmzVW01g099Mcz6xH6"
            b"qkaO8kfcroqxeAHye2JOkXeZCPS7PgJU8zN+0b5nA0KRE3eQM3n4+2rvC5CWb7cc0dppHuXKudgFF1Rz1fkQdZfDNqWFTwiczyzY"
            b"zvzjnxdFX7m07dYp7KDbpNXWn7z5p+XNmcBgsVeaYie5s0xIG5xGwgvtsHWGUONwiPcOjIAFYahySjtIPpta4DfAsugAxfASad3p"
            b"TaJXkSlNVcx3k/K00miKNYJXCcXCd9gUwI41cA4ZSC227cq8CFSphYVxLQ6gxqq2j9Vnr6CergvUDmBo+yWtNGzTzlQRsNnAY526"
            b"5mEAKY6q4mkXKaJBM6w94XdZBNHKo73eLjoyuEqcuFIem3m1n/UmxiZt1ebWkP3nCXFNfFV3tSTEBw2IzPoq7alRins5PIvp+K2N"
            b"+49lrRLES0IpRTr1GZAlmJFAmVYmUMWQFLEQNjJmMI+FUCh5Wf6e8vbC9/5iNbVCvDpeEu6hXqXmuiBA+AVIxFcr9JkMzLBReqLs"
            b"MuzVcrrS1cimDSnncmKFarCoVu7+9mVORsf1csqFOfiOiU8++k2Akh2YaUdV4/RnLKFJ8KuVmhmy5EoYoOeL29vW72eiXrSWjGke"
            b"7FJizXisERJrdcTqvdLOpW2o83W0hd8wD7P4v3r34/Vukgc2SPlkbIwweBQTCEVEKw/WGhlrXwpGA9dC4/grRpQ0mkskAVkufrc2"
            b"zXHVPza3ebWEzRU+7lUTIVbleolquPQ6OuQDPcprNkGpO89F6xPUtH6bF3CxHMKMqtvfpApVt2DJgJKuMTIeZuZAKPIziDTUa3qU"
            b"icgxOfWQ9TyWD1eHh+CigACetfS57Zd87CmzqTz7+gxuJNn3Z0by7LXM79p9nkv7XpNs/BOnucsIZtZabwQT3nnQcSN4iS31VoSg"
            b"hcIQq3aqkQxgfKzVpdY0JNtOrwi3hryDEfwSX4lHI78vY4dH9e/QVf6aWdqYITzSxIVDEhlJk+OWTbd1UnJeyrNaubQWZkcpK4Aa"
            b"Pw7Vjmzm1Q+ZnDfSc0TSkwBGGvExvB5Id3rpxMpBzJBp/CD2S5bYeK4Yagmw6emWyHRtnpz1EcIYTXabd7TAWghY+6h0PGTV/ZBV"
            b"XMEUrziz/ZFcOyCbS3L9qHZRq8Z7u4NPowu24YarkB/pivHlEeC7sq93QVFnBHEaK2EYlpwqJonQDMdfUoeU8Sgmn5guqBAoPp0d"
            b"16gR3Lt3IMvmufqfSzdMkuSl1DzucY4ZFrmsFfyIZD4aUK4pkmLad1hHbYYUlvac9Ax7va+lCd1SjUtgI484iIMlIv2sOIDjWF5K"
            b"6w3/Uc0c22pZFBwMogPvoq47jNZ+CxH8YEt0lHJkJTAlWtgJdt2UThYG2lO5OJ3hPfovrui/+BL9dza5m0/71tR7UErgozFdS9j4"
            b"Pmn3vg3984n3bTb0PH7+ZAhWIE2C4Q4TyrTGyZNeY4uYFUoKaoxAngEEySQL0nKpk/wyIr9vyDbSPjiCWy8MnabaB2r5s55pH8AD"
            b"pdzxbOqofUBX2yPyyGJj2MYWq9oZpngfOZa9lvNowWesRPbCB2TG5XhC/uCpedt/VgThPzVo4wx0XKFbHhfvFBPBkLagwCCDwGhh"
            b"NTexVvTxo4JjxjPBUe4MSHCAvJ77ob+hiPy4Qc/4+34YzHd0h/owx9/Yq748M2ZaK9VwnMvd8+5BZVmf8l85AbH2JmcvYdtifek5"
            b"LgvwqhL9gjVPpmmslpePjHnSIaRgotbp1qZds53A4GUpOp1xQ4bYMm46/e6i58ClXP+SAUOacd4jUdD/4ZoKjHe7HPgfrYnA1YOv"
            b"TNWONjn71ooKG83Qvo/rQl0+zugVG2jnIXPsoe/C+8pM7g2O6dkn8zPHLePaxXzMhSRGYusBO2k8l4hqsD4u+L2QSgDEzyENVv2n"
            b"qcBfEHR4NjFfoB9fYRVf7sfOlR+O1+OB8gM/tic6IlpHRj7OuvgmFUHLRerEyNHa1lCH5mx6wcGZYfyCmT/DO3nAz3nmQEPN7TC7"
            b"dVpu/HAq6vBRu/zEs31vAfSSDWP1hk81ZP+DfF+uqAHtHAHDlddWq5i/baypCdIMcW49Jdx5GRT1RGkkZFJIJIwqHzMWfq3L+tMk"
            b"3kFyGVF9X5hmB/Zj6GA/dhFKNnUpm5F+c2A62wQ/uyL/WiV7uad7Msj4X/M1GwLasvVYOlCKKGvRxROicooruz7a2gxtgl+TOzGK"
            b"9Qi7N9TKRSytNcUqthqvHYN5Dcfl/6tVcZpH8sRrrPG32dqpzWZngFNPtH50/qy8xpaS+U72fJPXmCBWKMKJ4Z4lv1+DtI71LqFG"
            b"Es2DRUmSw1EqDMUGpNDKURIc0YJ6Gtg3kpV9iOSawrh6LNP/KjsB2pzFkrwydGkL2T+ABx3H3KNgfegS2I/7L/jFL2BhSQ5oqAs+"
            b"Es2UXuapvuSLDONyoIPICabtSqO2A7Rlwlpp1BJ8eNtulzKb2PQ2NndbqnBvsDQGRS1bFdkLN+1R3Gk31l7pR7f176IrewE4daOl"
            b"KnfFJLQmTjpsqeZDe21LVRiIhZ2w8VNIQJsggzNBMekkFl4oKjDxOgBWSHsRgvegZXLkkKCtE/y3W3YdVLsKWLMiea4eRBWcM1m2"
            b"qp3V2Uf0KWZo0qUycIgsNE9RKrytv0BaVcU8xsfrmH23iWnjaI5heOh/0B7iRUsvhtdpOFO9eUFlzzTNhvRwgfOOcrITK7F4Pfe8"
            b"wE7rpmaf8Qrl0K+lupvWA7iyHqilB+YGAzNZg7Nc1y6RoaO6fhOA6IM89wAI+jDTvdC1S6JAeMLbc+NYzGvYgUQ8ftO4sJRi5rgw"
            b"QROsBQMljATOSJKMCdLYuOT9JmvdD/i8fIm/tE6Y9mptBKcfdg27mg5yRE/uOsYdGFAnelmzadPAxLoaNfH6TOavKMyt3F1klKLq"
            b"SuHqOyX7riJZwoFjIPtid3m32etKdDq8LwoHJgrUjeTX6r326q+ykc6WRyHtHdvZPiirXlmvemVTNO4NQnriabii69fk+bfife2K"
            b"V1Ibq0GCkbMGhJOOe2MJ9kgy77WEQB2zwJRWwgob/yMIixjKlQ8E5O92PrzsaXhltXptVThFoneBOSIHsj1wmZ8dQ9lCzF9pQHQz"
            b"4aLH8Jf6M950U5QLyl+m967txcfJX2a6UkIaAaxntlt8lwM6/E1h80o8kJzK8asDyUmi5/0O6a1cjKvS8ShTeBQgPCtE6YoO3XBM"
            b"uF608xGIkw6UVH6+wct39TSU2ZqQOEQNjnWDC1R6gzRhsWhVCuGYjlX8fUy+RiVFgph2g49LeqwERvIR/GkIfnoRfv4fyRuSZb1e"
            b"krHYdfcPOHEpa/j3EHheoN+bVcAwasdXbWqxnfJTWazzRdmar4gRMZqCpyiBjgKJ/AL6a8wgEBt8XWyo/yX9iTEsPwcSIVEjGzu+"
            b"RKgEEqHQunQHNIpMlzFH1ZimW8v1e7AkfFhv76AkWgFBd9WVuYLrDPTOtvV6Tffc8uWnlF1fAW9fhtUjn+3HMHbUrcPX1fj5Sr7O"
            b"gPuKXQLVjEvHnEXxo+scJYhbHIBJojwXDpRH8RZtGSMxxTFuuWUggMUalcv/srDryx1gHgukXtCSvapxyvICsO5UVmnqyIxP14ZU"
            b"Fi5rY6Ij+ZB0dPFytGKstYJsm6PExvRRzUsamVr0FcuaFYWUeVW7GkubkPs/WbosJXxAlhocVT7+hTX1JN/zXoNgVnrO+6bzZ3CV"
            b"YXGXx/EBSjQuVPfytJSsf+KuL8UQSWM1EioIFItPEBxJJqkEp6zn3GpiHFCLPKhk+mIgqJjUJWAk4iGKwL/Eq2f/AGl/obV6bIKy"
            b"Edb+qHI1lK4aq1wNYZaDtH8VwD+iC5xZfItDET3uEs/MpWcqVxs36lGPV1X09kMr9qhytSVVNlwpjFSuqlQ8Pqxzlav95RdVroZS"
            b"aEeVq2ew/PfBTXNQvjwB5csTqSt5KnXV0PG7zZkHzBJSSV2N478P3H/WBq5lrGZt4GUhReo7Aa0BpfQyDeAjclcqFuSMUSbAMG+N"
            b"8MjZQAARo7VSkhvrgwnGGk2kd8KD00SAQzHrMInVOwRXPs3jH43m6JJW0095tMMivcxgGW+QdsC+je27aFXuTcAVkkfJ/85kpiR2"
            b"gnejmcaxod112mG2bRCwXKdGmYS06jCrMQ3BjTVNU6SSc4jqEOpZ9lfucAwh2e+aQHckOSjfFglHBE+sI0gnVp5D48+K2pVeoir3"
            b"CPKcuHoJKRcRK8rIykpbTnx7I9JdoBxbDil+qxA8hsWWsHhRmzvH29osM50ANu05syUKKjkXqPliA45uzetquV78bHOTGTi2aX7w"
            b"cmAqhNiwDRohxHcKwNy4M1gXrDTS4kBtTBZS00TcZVQj5pFm0ntuOQNJqCXIYg1emuDitwYb7e43stkUcHY6VpzPFMV8pgiPAGfs"
            b"sTDizKaHjaUDOiWXHdd2aAYUURa2tXPxKlgwjG6JFWUERy62m8ZauGR1gsCoIV6tYlhdqyN1k1MkEXK/nakD7jj1dVIEkRivrZpd"
            b"E/2SJky7P5nd2jOAlu4Nr6Maen6+gZ7dKcmpfJVFWaEP1DALvAuBT/hdbGUa0JppUD2X/n/j2d5vZ39YoVAsf8VjNxsv+MZjN/uR"
            b"FuFLu9mKcUAxpcUijaj4LRRAdMyAHISyXsbPEKGKGOS4JQEJrVMHRTlFHFWCIvZfZsi+nP467mbDTX/ek242XO5mw+VuNnyJI3up"
            b"1TzA985bzUeEb4laVGRRRU6ra23omvIl8lMM2b9e9h8l9rSdrQQoqq0KjhlFKXLOa2GUNPEYnNSecQmMKCUtoiSQZCNJ4qYwyAgs"
            b"0G8nRPSCskNCBHpIiEBPFI9yY/mjAWf/QNHNcfEHrd1lYBKe41I6nzIZUEfPyIFEMtZKZT8Qth2yPfJe0sGqskzfcHQPJdKPCr95"
            b"Hyniyao09QpuEb+eVgKA2jAcrrp+VQ7hLfJ3F3bdRAW/u3r2FyrTlzIjlMCx1gQpSVyCE0p1XGlTRzzX0gFyxtJgMUZGImqpx9wx"
            b"xwJD4IzjyNhfIRHYKvtdkgh8XtSv/fafivq1x/UqUb9ex7CI+i1JemFk0LUubBmtMSa1jkntALakVdHK+qULNxEKfIVE4L2V9o3J"
            b"16z1KKfygKV1KSt5QFlpW4/lAeUOlp1tLuntuxu1jNRQ3266cqAmKMOQFpLKgJUEjj3hCIPUJDitNDAW0xzXjhuEJYPEUhDCCOux"
            b"jDWcEv9pAaoLa93rK+vHqlEXEBR8yXMYoJ5NifXwcZu/FttECgP5p85l+0gwqIZvB5ms9Pbp4mFMj2MscfTRzrstIAg0mtcdLlcO"
            b"TBdipvgvDqLSfDmcuAxB8ihD1Zk7UlgCP7XClkRKju61Nvkq9L9UkdUjVmwNN/RB+4gtTgJb9dg9ZAsAgW+5mTfSg1U5WTkfHp5p"
            b"TLYOjlvvR/X+F9fhjlvOQTBpqETGAyGxQuVKepdKTwwmVqHaeUXS+Mg751BAJqb5uG7XUvw+uu49G8GLHdPP03Uz4RXVOiy7VTbu"
            b"DP7uMHuLql+ah3I6nPK3C+kcRha/F9FGH1FeaIkFxhHqQV58zZCNQkwJ/VImzR+eu91KWnUr94U4/V8NyWo1Bka+szD1nb1Lx/2o"
            b"uP9/jpCrkVLWMfA2/isCCZojEMn5LVYu2DqOhdUuCbNqMFTzWPFCTKZOWjDB8g8t3e+Zzp47zn6VRzvmQZyYpjw2BWCdSdVFC9mJ"
            b"2l53BgN5wHLP3jJb771yW54vpW1onQXV8UreNKZNuyyhWWOgtfiemMbmwCJvyDEVByfv/kRTTM77a5a9awCbgXh3xFJpJfFHK0FU"
            b"XGVV3GVYXJefuFXBnlmrVEl0PAiqAioi7kOhv59o7fqPDFy3DGvixzImTm9YyqwiKOxoSCYpCizmmigSP+sSA6HWoLjgsk4zAc4i"
            b"FfyLZWPu4VfPwatPVKkXisX3ul0vCSWv9uMWGjgw3Ww0zDgYlwgYz5e25ejzjQkYRfJ4Iv0dqoSlhWcSL+pq0FF2zGH5rjYsWh+X"
            b"q29Dip7jQUeF6hlVd1aoNmilLV02XtjVbzuAKKe/DRf6j9CfaxYFJrRQzDlsgMbcybhz1CkkrfSgYqYMlHLrtTae+WTxwrCX2lDG"
            b"QvD8m6zmX6iqdU2z+ocJTV8UkL6sR/1LhaZ58hq+q9E/1JmGE51pONWZhhOd6feVn3+L/KeSJ1USe+WJMmAJMgYCNjgIBDZptZIA"
            b"HqQjmiLhhTDOC+KpBOxCkktg77C9mnhefYWGe87BPTexkredsABtaKJNT3nd2U1bKHndzuqu0VbaTwpQeFWwKeQgvAIwOz+qFAdo"
            b"LSdFPa5qwU64hMU7x1YZLlWq7AHuGUoEG6z+GJWfpp0/1T166jMzo8aQpIGuD6Sw1vKS05rHyptJEl4nOzs7le+vXR7lr3r/qCop"
            b"XzEP+ncGU5+1kVrTm3HBxtIQWRmYds44ZWVcWlsnAClKtNCJOUqlZIoZK3ightmY4DwKBttvJyqIX6k4fU0IcC49eIBFtpj1HaXw"
            b"CLM+04ouGPFSpUnAT6hA45EQdIfQJ0OMfnwW79DSvXfZRalVyIVAI2i4rp9qpBGB5TyAw6p/KKqid6SPC1lXEA0UAhfN6gtCiB0k"
            b"dCAk+HYNf7yq+G/L8kZ1v3vUbA91/6HR+u88ADq51lZCcLz1EzX9K5jniI35MbX/AQlTe0S0AWBxcQ4m6bVKxGyaFAXniOMeEjZK"
            b"JryJ9JgBOMYNRR6Y5U6ZP+X/P+X/X6D8n873jijKmNwjp9L/ctrtlFPpf7nRgOTGvWz8nP6k/+8B3HVA2EpCvLWSg6cMqPFxwey5"
            b"0c4AZQrxQDxHjmHHKSJOemIFsUgTzNl/WUjwgqrfpSnLY1QoyRPQfOmq0dKKhO87jVf1Da8alF4WQvyMyN8jS9MuvFiNFqYmY7x6"
            b"SfNWg+uYo2m6F3Co0LN8XegPNMZyZHZDfW5alK7yvXHRGNdUtFhrGSraWEpDNRSCaihEa/8CWmNFj/T2+eyoFhf5kEHpf01ZMBei"
            b"lgSqOGBIEySHlSdacwHMCZm89ZgxTDupUmceQtwmNCb5oMDhD/nr9doe58IeZ+oadBT1HMipfNnZQ/hSc8zn06Su5TDFJe0coy5H"
            b"DadJqOUlkUf8qsFQ5hqOajpMEls39+IwaXiwOY6kLqhQh50fBk9JkXUdPFXL+z+zUvkSDNNHyJtiEWU4nb4fvP3OSZ1vGiAZDZjH"
            b"9KiDt4zjuH5H8cPtlQfHQRONJYBAXtngYnVMsfQoVr5BOCkZFfodEq1rObs1uG9TneChcutd2dbrJKdLI6bhLIgOJJnuUZIuassO"
            b"/KvY8oUYH8dhyDWBl140iS5MqPSnIitcaQMciB1w0BxC2muJh3i3gJa7yuavK9HZDIsQ2F4nW7RpCzVNcYtSST3Lvzu4ulnLzrPw"
            b"GcRpUW0ieE2z1SOWMy7BFfl0e1BLN6291plCHs7zr5cBoD6hmjqaYD1kNL1VG/W04WosU07FapZYFJJPgVTYe6a1T0R6kvwNuGXW"
            b"Gu2408oaG5/iThEw1DP5jjH+DQuXl0s4DbVRr9NMG+ermdrmVkKe4ozyXrJEqtyU/mcoo1G2nqXqMRt0SOvs707jSVPeT25AEL73"
            b"cvlc8qnMqFNnQOGBelMTnEIKERVzNFBuOrQbclyFnHo7bordc6Qes/hhtbaq8Pn4gqrTuJ8wA0WNplidjNO3Yo02fYEJSuBEYvQu"
            b"a/Q1+IF8fEumXXKsBcNMejeElZQSWQheGE2tVwa4wEpbg00ImEhhiZHxO6IxU0ERhHUIv9ug8EIr4bKH4TVcwkV3wJu2f2sJvAnn"
            b"kyEToaqAF12Vc3/Aix6FV/sN4wK4sxvsIFzzAng99PHrhgWwnN1GBrXvTZvCm7ax592HWd0rT+teeVL31mOzte6d+Qtsda98XPf+"
            b"WRberXkt5swllGpgMTPz4KwmVjriNadaCs4RIk6JxK7SIgRqAtOcCfCgGBX0zxzmzxzmzxzm3eYwC/r2c+YwtDaHobU5DG3MYeif"
            b"OcxvNYex8dvErUDMSIQcDYgmuzCNgKKY+rUJYEAaqoGDRYFQDF6DoNwwF7zzvwz0K66jdNH5BHGX6FfP4IMv+o6TPaHTgiqr1FhX"
            b"Lf8OTlxCAStaK2Ntwv9d6yJFpe9nrTs9UP4/zghHA8IxlBgdlKtzBYcwWqt1tSvCHF0ncyDE+5/Yy/T9Bc2+U0yWeexq85EEwqA2"
            b"vzcaLHfk26U51KU5VKX5RAqxFOOyrszlQzHEe2V5lcy/k9PAl8ryAw7uXQPB8+xLsXNcxXTLBQcideDJogsYxDQbMPIUaZ4QFwKo"
            b"xsYLBDErC4Got/EwP+QkfjY3XEvHeS5+aV2/VeN4x1TRqfH3eByXU7VcY8okjl20OxiX1WTFYGBACz9s04ZhlwzR1YIm63HKvSE6"
            b"HII6z5l9IHtE/6LiOiPX1opYMUyjA4xnkAJ3ZtoORGusYTZq2tNl791EOfMRP+tg4NpxBbdeLLRirdXbsjJfqVPmnZT4bVhmLa6s"
            b"Usy8yDN7oQGLVS7eiqVhCEHMbhQJrgzCHnGqFY01jWOI5zSnIBCsNZJOG8VNTHpKqF8hc91rtRbdVMxa07pWQpQvUqyI0wazQXqp"
            b"0QxLzKvxGrQ6HKNd0c7Ou0nWTlBrTatDI2KiytWoUaddJMQaKFjv4StfotfjHkuBPezy9O3f+PrM3EW0fcPj3vJBpbhG3vo2k+Fu"
            b"S3aetGaKKo9arnCsHHtJ62oBXm1uwqv/XLjqBsWrl6eKf6/iTj8QqXq52rV1SMuANQHmA1HIWKIJVwhhKxzVXIa4fOZUCJfUqShH"
            b"zAbp4ko6mJgG3YegsS9ZOE9XzeMCbWuX1od6Pje6i7ZdWq4VFUxu3Ku2NRtjKjWoHHeUdR0OncTtRblcFQ1gbzXK9YaA2z2mg8px"
            b"B8uRLjA/n76eihf2BlsVUnsXgxSQW6tfoWPddR8pEnyyki+V1cBeVjlMVqN8KNmmTmKXvEcqkdPx5kvy2L+iqn6WkLrmMW+MV8wb"
            b"MIFzjZ2XVhrDlQgII0O48Vhgqx2yNHjGmcTee8MRDQ6c+1PtP1Xtf5EY/2VU1jdR7b+twn9f5j/tuIjwU9b0M5cE2cvw56iMrvqe"
            b"ov15LXxTsp9UAKnm0errzGsPFF7J+VfL4PYRK/P9fVVM8NGT9KjdNxXx/5Pq/4RUf6xFtdFKYo5CLEEclsIyYSkY5RERRFJLCfPx"
            b"c4oUNorZ+NHGLKZ1CogF+7NFqK8pR1/lqxZ+VPy8yaVnSiviJnlSkPo53NcXNKsvi0oXkeic2glqlROWfl5/AHzjxcpD+2pwNdNO"
            b"oYBasahBWKICH/R/qxyb3yHmVGD71RWjq1tiuHxahfqmPdV8nT2zK5X/g4ocW2/LhUILFYX2Mn2r60fSautPgPp15C0HhCkqbMyc"
            b"UjvBtQUEATBKKn8upCE5x2AtkV5rL7lxAnOqmLTCGMO+BKGaVLz3ZKjvKg48VZAOBU/H5ehmdQ8dKqgMBAiWRdJkOcItnPQVY4kk"
            b"bBHcq4tGAm18CSqQKQZE4bEMQD+GWStkdMx1nfbMReHViYB3iyrKb5juQmgFrm4cWXI8yjyrXnL4Y0nudIz0Gl4ty2nnXQKN/zZo"
            b"qNr5dXDlcny5zl9CUr2LnDAXIViksngv8PrIMGCmMjOiHWxchYEIV29Dvau7fsRw+tOq2MequtHn+rxe9mBY76RSglKlgrMCWawI"
            b"gvgp9UxKz63SMRsKoakyXgWCSaCGGoxjZa0BWx1+m/xrNYonyzCe3lGHzWtKnKuiTe91SAujC3khEQhKsPyinCsvcjSS4ANOtT/J"
            b"+KYyp2CUYzcTgSlo9prq7aF8PoHhprfOgQ9UYl8wir/t0NraTNeT+LH261o2dxP7rWweebfMBlfH3/6pvV5Xe03SWsoZ5xiguKGl"
            b"EBgRa6mOyc04SYSwybqdWWKI59QwLYMkXkCyASS/0Ww6fWyqz099Fkv1uA3qxd7Y5kOBwZTc5KBoq/OaXAf6im8An0ERld4txzwa"
            b"wF+xcE67yNW1XLWk1yV+v7eVs7rUyHtLtW26qFTMr8KHtaxACw1IMbiCkO4e1Ee9hnRouJ/D37Dru8VOnSuq4oZd2kbNFFVPlVun"
            b"W0WJ9c9f+sqkyvmYr5C0IEOsv0hcfTomvfdEMMYJNQYMoQLid8EJFXQIQQTtdSAxXAL5VnKpYj6Oh0fjePYYyj6TENxnGOTwdZ7j"
            b"2O+TWC/rYV1FxWfsNt+ppPshwqSJcZy81dd8rk045HgdCE5Q9S/5oX8pqgU5Kgvyfe/bkR9IoSm0uKhyWBog5VUb9bRR9jrWq+V1"
            b"wAnnh/Om3eQvBz0vfnXLJ2CcT8+emeXTs2d2fZXHW7Ta+rk6V18Ctr9pFOVjegicGCAgPFfxa0CMlJgblej98V/kPHOKxaoSS8oF"
            b"R9hpRFh8DTbKvYdW9Ag38JUakz8YVb2M/3+NKNR1YI/8fxjznc75/xPT6vOO3oD/Tysh1ONRHKFYe4Z9Pf9/y6TqHv9/+LoZ/3+T"
            b"YyQPOEY3S93/puzV9y2HH6TjzxTKg9alj5mYqSADd6B0SMKDmDNmrQ1Ie+STPgDCwXrshA6Ie22N49hk0z8mXmv7dyXfkkc4rROE"
            b"1qz8KxmieddTd79OBzo1DNm6J7audUsakOeuesee5sx2cFLs8npMlElVl1Rg+oQjVmgqpktQrfGtjm7crIQSsjY81gwL602E3ZsY"
            b"5j1lpYK6S/lPBagag+gqasSchEX2GrZnaLvE5yebyyD/2+OeTlxL3+q9d1jqe2KpVdRb73SgnCPvg8Txg+M8NtpSBiFwHVQwMVLH"
            b"XwUGnhurBTfBud9tSPWBIvKhiNTQMeqrReRtEamxb9VFf6dXikjdLyKHrzsRkdrQxQcG6oSu/nb20p+g1H/WuMpTjZTWEKj0ihof"
            b"bPwQeumQict75BR2lCTHF4eSk7RGFARnxlGsGbHGfmoeTvtUfQ81dQ6Zuj4P78YpNbCdHZCszVAFb9OZQyhb5irV+6pJaPp2lWmT"
            b"RDvpfBk3H3nnKaoeTNXZp7X1S0H5BkIAt/b2Q2UNXOLij40dpdah/iAasRK6lqCsAgCwc4eGYdCwAH2Tkd+MqA7T8ThMvageDpfG"
            b"lWeFIvo+iKGS21DdyXwwYXqE/3npZNxrZKygNNjALRMkVp1YKSODESQk6KcjClDIfiexEpXxeKynHAthQRj+N2E6mTD9i6nRlRnQ"
            b"ncnXnXlRb1EFyzGufi/99GbkNrUDUti8Q3BtHFZGUtkHK2ZVOhouTQ4MLa8AghVtjw2Gt5tFmlo9761CQVGSILA3+5vQ6Hc0VlQj"
            b"mQ9WzFNgd1aR1auKVt4CCm0e1N3NFadZfrf0MavNeyZVfwOnJwZOlmvhrZBaYORtBudjKuPdHicPKsk9ieVpINQoLiwhwWNERCxV"
            b"U9Gq8O8WnP6qefWxlLyiJn1RxzmvqpcsyeUejE9klnNkbghghmXtDr1SSyeSy61dwa600l+LK1CqtJO8L9aDwQ6dkxRTUFeUDMhV"
            b"fXiKoukfsZ6aasQHBy9In1AiGZteiyacseXKPSs8nT8r1yth2XCdag6UbICe9BKSnlYqd/V2o/xfbe7Y+dYO4Gpf9k9u+m5rIDBH"
            b"YvINyEsXhKKGx7W/pcrqWBcEQ4OjjnGsCNWaK6EdghAPTnLpCdY/jyt1psF8TdX5yPC8IJZ52Xy1lTGtSDvN/liufWp76ooY1QsM"
            b"bNLNjQ7zzHx1MBQbOpSkE04HICR0bn9DoeapoPVQ7Cpzy8r8qzRX9vsEHR10ObkcPNabHkuH5kPZ9Kb3nogcmcD2Wlt0eU2Sn95O"
            b"aNWsGotVS1iin6VM8Xi/YfTGUG3VzNu0W5oamdO1Dk+P20crnbWuqEePWgU+erDjXgQFeoJspUDQUgXq37+YN/BHpnqGTBUSJkG6"
            b"5D/gHddAUp2ukGUyUBTvFoJyweKtwDihtFBeceBag0SUG0v1f9kK/LJFdncPGdxBrt4Z0j7Sx1fte1rK16duSEf605mjdtGl25rL"
            b"M6vu63cvhvfGjATcOYLxwywxv3cC/4KktXIqHaF+c1C+0xDAVfSoMZXfPccBX9ox68qgu/xqiXjS6vuexCosjNZaNqZlvh67KKs1"
            b"F62suWhVlbfeA/UzPdN1XsNzWm/+mXy/tIESDHPIoyRmqFBcWjtKlEJOGkkUSOasxJBgujZon2y9qHTSx4+jppJKLX5cyX61Dr9e"
            b"2V9k9R/taBtVmUMP5ar2wC2VhbS7TSwAVxxUXFXm7Zwv7bZcgXjP2C7+XvQPNafL3nOnJi4IZTMe5GcvS8HlfS56tlwAn1228lm8"
            b"X+JP0WhLQmvJuV73FJbe9kMSB+nTc71Sn7VR5HTUKKeMjLNnCoft8dbagum32uL8ryL/fEUeS+2Y0o01iJG49OU0Lgw1poYbzDWS"
            b"mgbkfaydrBZaBREkJkxybK3UnrFn4BzspLE+76o/fauAB7cKNlPC2SBq+Ut9/EbTcjehsMocbmi3Vk+w1fneVL7hLojhKAVeNTx6"
            b"eAYvyKorWI5FOrxQjWG14J1jOXIUZLraEl15qI1fEAPzZRK4vGS5Uqx7TXvhUg8h3allujqtPNxRzxLLIu6wcOfYaYsn7zGFbrd2"
            b"WF8yPJJ82Dn2acrGbZOvteiuC/A1G8tDK/2K+Vdr6dU+Spm6duyifY/lYPDVRn8b2sYtlMkXiRtv89ENlgqnjafIIYY0iV97gZBU"
            b"REiBDVUMI820iamZB0+F5AEEVkgj8PEf9bvHmlfnkK+dfl4DLF+1DmNVC1si2tw+6Pb2vcb649lAtsZKH2PRaj9WR9u9oOiGo+z1"
            b"tStGblrpI1FHsWGOYZ1riiqDjm5cKRCKXzkig8lwN1VBpdDnBFT7Ieni4m85fXqaSfDtgvqIw5NTMYn1GdwISOzPjMQk+pL5vHiu"
            b"t/7mmC+bY3JEOEtu5VRL5AQhzIHzqReAhZLIWu0kIgEnWh1owjQywWLhjAXFFPXvaIq8Tq1c7nrlR4LatJX73DiTLo2U9FMelcNJ"
            b"b2hb6g2i+Eg2vI9WpWQGrpBsm7lJ/rENTiEpe4IATmsxneUF9Pnp4hKWfrLadmcczumuc0m6ho3cnUK3W51c7nYEK37URi/VSvuK"
            b"FFfa7PFzXKHOm+K2FYRI16REM9KIT6iTYSdbgjljCNb2h9xV0rsRZwkj3bjyNgfwnlJZaUhD1ZyGqlKGqlCGC0Y6c9nIowTvuX4Z"
            b"HbwGGu+xHyyTXtniLoX2BVvct1IMrzZDOGIcSLDGM2G9Igk/qImhjgvDFJdSYgxMKq+psNYpHigiFGsMRlrl9Id8ffaiGh7V1fSs"
            b"rhbXcnrvPrsT/Va48gxOV0Ly/hQ8Qst9BSPeu+GMVgib2Pg5WHHrR9ADXrHz99kL2KNKBSm9AtSOA+sSmV9Cl5MjIREKnJsgTlqY"
            b"4BAjWOJkXd03B3Ks7Mtc8rmKOv+B7ujwzrB+c3zgnNdNKy43bXjdlQZvN1s8bkI2y6g3v3tl3SK2C/fzkc/QGyvrMzV1jrRlOshk"
            b"/ih4zJtYWWF5EDgA0QhzLSVoobWj1FhrNA2YKK8ochYzC+8gD87r8FcX4dfkGi7b/YwY1/c8efDQjWfYij5HVG83ADXxmTxddOS9"
            b"5MZEuaGc1c/lBDLFhvAd88HrGrqDjC8MTEzVhvmoyucmOIWI4lfB98s07mkvTJfS1Fhdgub172uMguBVEhglKe7tB9zIYUBVFz9W"
            b"xqzSa6d23sOv1+bGSlP8PlVuM9ebKPeezPXumgG9q3PMkU0CmAisd4YiHpAXQjpwGpNYoSrOKJcaGDLeeiw1B+1UzMEIe6ECo7+N"
            b"nf1d6M+w0Z8vzNd6laDBeeFCASx+5ivqZBDXahzRoYFO3AnPQ8UStexvQidPJ5BCvwsxG7/KQnwmXD6HSzzoCPPhZrXc/2NmX2Nm"
            b"c8Bac08NSGpwcEoLbp1PKhPcSWSssTHFhbh4kYGxmM/AWS+Fc8hjFzPhg5w2zGgvKhj/EVQ4Zb00KirVY0ljcPQfLxFbj3OTIUdd"
            b"oZd+xfa0NYjaXR8W/FtP2MAZ1hXPC1XWPtsgv+OYpCiB6KTb+QB5PMToCbRq+ArBJ94/bWmdA4mQaEud0+CEhEiBRCi0ptAtgbaI"
            b"4iRrITqzhjfaNMyz3UyhZ+5hNm+Y7s3LDvj1TckW02y3lHRdEXgx36EOdYuWHHeO260HUnjLd8wxjB0BMDHFxbcTilgZEx6K1Rw3"
            b"BgKRDAJAXD5T75U11BLBQZAgJOB3jJ9eIt74YLn8UOfsWV/cI25qB03hc+GGGul6XFoyvC4tJRpmlbv6kHTZV/pJa2HK8T5LWLe2"
            b"HVs2DNa2Mz+JrC+R17W16MNWFJJO/TFd1FIJN642w1FZoXU0MhfDyJGMxHPLZSEEhnucZYILqGBJb81D2B9uBrr1Q1iMeKunoX6W"
            b"F1LFArtqH8EuHbF6mS8PCshq7UquShJd43IDaP1IpclX+Ou+CQ0AklOlXALDemmJFhSD1YpR4k1M1SRl31hypoZmCASktVZijxD4"
            b"5KaO/nHP8tNsinOSREWROOiFX9KyPaqtdSOSS6vOB5l+oISDljV5s+CuzIZ7LcecNyrvsoGRcPsStnVUF+gWa4wWj0a8eV60MtPO"
            b"bHhFZqYt6j9kvZHjSp589+Ed6v/kTgNByytlxUk+Hlg+/hL9JGntthZlaVrKqoEpqzpXVuq+sqpmx3y21ZGXVo68Rw3Kszp3Xelf"
            b"1aD81mv9L7LW3tbljOmWScq5IbEo1t66wJEjLhbGoNNM3sTPvI7ZV3GgwRDnPEo1sxCeoqD1NwRovYCXJp6EfE1gpdDDSpeEySir"
            b"icWy6Qxd46+1GIG0w8JMI0xufQZ+YKW1rxtIqS/peHBxT/jOF7h0G1cNdwjeVQyDbx1WXB1KKWAaxfkcmY6ZKolqxnJ1G+lfkgLz"
            b"ATFQwNsJVn9RyhUsgR/FZcnXeALNpk/FKGJvTLTPjFsWvfkE7Pvpftd7Ao28g/7wWP8Cj4UTTsAKjxQJLGjraSyodcCWBqyJsp5w"
            b"FZSwGowGz0FrZIQO1gTPDVV/eKwOYjXhMDwFnXohXiu+rqCfZMO7LR/R1pM8VVRC4r1/sDEVDrizFHYZ9pRWBLHS3YXU2PptxQco"
            b"WAnM8B5gsLTY2bp4GMazjTKBi5lnd217ffocthAmnkNi4VspeYbEYlMkFpvCC9gUicX2epntSCy2l87N5lZFsz8k1uuRWBg5HKsX"
            b"CAIMIsKIZGiOPKUOK8uxQcIR4aRghASNLNJWxpTKFEVca8p+mcuGeIo7dpHsdUOy+BrDq9LkWQle6pGmW02kQJc8yVGPz1VL/5YU"
            b"gcpFy3KJ7CZdKSatP4hQ2yHsdsjXhHlKI3sw9ctRBdcldqpIUbwVo3DBl9UCqqruUgu1Ry3UEvS0OSbjtwV3oBLcuQJznclgzu2L"
            b"dsGd1oeYdkKZFXzr4djue8gVX6OV3RQrflMjGSsavPLKWqIMloZirVn8omBlDGYQsGdJ5cxp6bnQSCqmFMPcIyBaO/Szwa/XYKWX"
            b"msDXUKWHBoG43nqZzBGvgV5rffa6+Ys3C7dL5kkzKO89tGzaX0HLMr5DGcQELpv3WRSLmdh3XvdhDn8ygZa9J2nh5T22YeF2zr2l"
            b"s8wdpfSCLw340ifqbqf46PxWJBVq9gE0/eD7RDE8wH+Nf9vSxegfJPblzeKYXpUnSWOYYce8YoYT4pVhHFHMiAWrDaWGx/8wEcRg"
            b"k9p6mGkbS+CA/hWcAh6N6x7N6tQLqQf8VIKBX7P0xRNL36EUfDef6vc8kYKvZXrVEX0wFoE/kpaLSHr8cOBdnLEGYhzuSBdUyPLe"
            b"SgMdNWLwsJaupGuLowWvQZA8kOeOYzeyRKZ8CrUcWa9YnVuuz83m/lP6799EcuwNc7k3VrjGg2LUe6qkUdgTikwg3HOCPCPGoFTu"
            b"Gm21YM5IhMGFgIRkSUzS+jf0a5dpzBdUw84lw54Z4d0TL8/42+zfs87uT4kGOawM3+JWQ01YZ0b3mrgTu3dxqrBY/0FWfENp8ZJa"
            b"4H1AcU0RqR8sSa9FQ47vm88w39Ygpkt5PNmDEk0OSwUxowJ37zBy98th+V4XywV0kMdpYBZkCXpWI+xe7+BZ8uwI+Lsk00op4QZ5"
            b"tvptRZ79IXbu9wQd327nft6y1chxEEBjYpXKWuYR50EYF6tbgYV3JGVVr5nDnIGjBjknCaYOIeqI+2vZ3mnFXm2Lvq5n+74G69VG"
            b"9akdnrpucXdwNuYbWnpRA+6MjbpbZFFHo/IjHVu4l3THMASYwhDOHDxppQlGG62wnSjbUmaH9WwV99eufWExa5jwjENC+yafCklZ"
            b"MIQJ7pgUyGCPpQBuucYh/h5zwSnEasEqSoJ24bdKo1/UO0+sqPg9wAV/1oZNXYAer6IvW01c1m+/V4/ftkAqcenKU0UoHZPPDtZN"
            b"OTSf4qcsiEYCNHiEECn6M30hPwyleLc14ni3NWoAxKwr/kvskyLpBN9DPdCqhKYV6mHG+j2DqM34djXorNvi/eYUitagI/5k0j8t"
            b"k86xNzYWJAoh63xQ8UiswbHwjqmAC2WF5vEmoJJpmMQ2OORyQPwsKydiSf4GmfSJRvpHWh1HafRx6mfrs7Oks6yti5WxPHIkniB2"
            b"ZBVwshWlW6bsV/bxPbO4IkJ41W5p9L/6Rsw1fsp42TDZI1oC450XVaQWOHRXUOEMw/7XF+2NG4oYPXui63ATKTablLGphgL738Y4"
            b"rtnHG3iXVlUzbcG7Q0zY1Vz4nYXHPygvviQwQjE2livjrEYoYBWQtBQHJwQLNN54BcZgMCUqFrAipjvEpYnpKxCOkIBr1exJKfsJ"
            b"tZevVrDX7Xou4m+P2aGbh8l1vtWLsuSd5VW22sZVa4rCozXzpeK5hdmuZw+VHs+OGyA5sALQsqJrsC/zyROF+UBGEXcctIcyiuMD"
            b"GKgX3taSeRVkdgwMOIPMzqAJ74PM/shS8R8VhGsOtbG4M5x5hTlnAZiUMlZ9lCAuqPEqueVgrTzH3HMlsPGIBauz66W2+GePt57D"
            b"+4+die/l6ZPJ0zXZ1COpYZbM+aoTsxBlTyvI4UJ4INrKy+5a5dh7KW+IESjJrtx8Hk6nRDWdEqSxKxus8XMMPD/KuqlTOF44w7Sk"
            b"PJt+zZbuPfb1QVOV15t/U6xXJlHHEycXwBESOLOOMUINshZJQpQAhrgjcQUtEuWLM24Dw0yi+A2mUpgfUYi+sJP5b2paVLBVAnXt"
            b"2+krJu3DLq8c0iu+m17xIFcd0utJRZnz4OKAifhW2fKlR0qHWCyByt4Jwctb8F0GYfySvPMc/2Q1+gSjdtR6nC/K5/XrvPnZs2Iv"
            b"tCx/p7Pjv61DqfROO4ZNXNCDQYxTQqzAIMAHHf8DrogzzDFLObLaamITQVYnW1ag5FcIhF2kCjxOxBeFuQZWj5uNTYMeTSEy/SOH"
            b"3KvH3cIvKWfl981Tr4Ua1RndkDN/RbzpnLWBOSD9LSSRB1nG9uzTu+aoxzJjQwHtXargLfrZ/J49wWykPzP2Ohvpr9VsXdnioSr2"
            b"mXp2zdj6E/p67cCfWu4DitVmYMQ4F98bKYNtiEnUMIUdFuCccUIjZ2xSm/E0IKJUCDwVqb8MX9Whk45raLzcWK+k5S/AsDp/Ftly"
            b"Hy73Jo6NR7KgPtGIgT9iZKHd4Hemb3buDgmlSOaqxYod2yh79b2UphsWa3jR0x7LgVZFbduPGF7cHFxEvhZf3v2qLDIG3V83x5HG"
            b"CBdvVPc7c3rCGKX3pvWZPrEpMbaP1t7oLttYPyot2U2GsXlwNp0qIoynzdfFr2vw4LsQaNf6tsrRpXUwVyaoEFmjgfxyMyjdhyMu"
            b"67052isniEKYeElUIMRJxDw3zEsdgCfWjHNKGQaKKOcoZ5z4QLETWGntTufwMFYDL9/pJ1L3g1L5jPl10rN95GLepqUCTq10VdTR"
            b"VFbu0FQ6iOhT8MjuatRg7T23S1bDK/tJ1AT/DgabgqhSD5UAUkz8ixVLWVXhiobRRYOcFVxUiV4Baa0eQdxfior3MnLAaz1kLE9u"
            b"ZChDV2PIk2P6W9Src9rqmJQlpx2BMxOEXfx7l0e8Tmn9Jt3TxTR0oPv9uEt6ru+NupxXK4RvKOI1tzFEqYipSttYftrAiVbW+CCI"
            b"0QIjZJUilmHruXI2Lf6RsekBspapuOR6T/35FSQqPECisgkS9YWjpjEQdPgVvcrfulbE7ThR2Jj3pwXkXm+i8tZswvVcq7306URt"
            b"4l76tP2c6Hz89bji3jEKR+buwBRR1apW96Zboxo10SKWsb6A7Zaw4RbYYNGQAsuF54wVoxo1qclLgNwBp3cnXfEzyMRd0RdeC85y"
            b"WrUCON0VwdtHsKjW8lq2llfK4oXYtiqLV48KR6tvIlRq49XmJiM+lBevWxK/ayj2oOb9/LhsUPwy4il3RkqEFQdsmPYMe+KZpUIZ"
            b"6ZC3FmtlvDbGebCMCWst6PhbG6R+pvhdUhCaoVCfbhRPq98XtoiXmm3rmO7veWj9lsbvoeubp0JtG5cM2Q54qS1rUdetaGwotyy/"
            b"S7yuEvX9WXLY72peqPDBDUdctMLp4xRecjbm9JFpRba1SXEUUfrAMyKHULSCy+gqAdPM6kSJqP1v3myBODctnLGy5s/A4BEe8rBG"
            b"LdzvLyHbKAqUtiy+0ZZ9ZSlMwREEksWKOJY6DjErATtABKsgPTOC06AFY9Z5SalR3klNBIk3fmz/lvl/y/yfusyPf8g7PdHZMp9N"
            b"l/nPDf53sex1mc+2ZT77W+bfzG3xA2I0gA+EcsMl88m+wGDMg0ReOWwCBKyRdJbGu6qjRPkgYj7kDhz+aG57NH2ajZ5eN3ca5zZ4"
            b"mNvgiW+v3Mbxa+6onAIO7YISF3/QK4vSHJcSE2GKDzMe9FoAKZBIxs4T6TFxH+djeS/pYFVZ427q0+JU14uNhk15HylizXBPDYDS"
            b"cvtWIcdr3hHvSEjVJKh5UAhKsD9XPYB1KAT1UGgr6PZRTqtwUrtvfUSW+usjngfk+oejnFfmPgFCEBkXAdhhYuJagMd6zngmnMNa"
            b"AKPOEOE99oiEWPZxsA456zU1FhD7zV5aL5bouyhWeolVeSpP2r9vjilselQvhgfFUg4p6qR8T6J85sGVoxbDxd3vcMygx0tQ+kkv"
            b"GFvfNdRK42fYfcIr5ZXDnvMRyPxlJ7BJ/xX86ETGr0be31Tou7U4ni9nZ8vm5xbUY3zTI8zTn2nWS3VQk1Wh0dhb75xFjDAel9Ga"
            b"GakDYUYZIRWigWIlDJMqBONiicqFoZrGzyL7FTqoTwHzv5FU6mWVDlmlbH6Yfzy6W8wprHhxJcEVXnZ6Unl3MoPnSY1bTX84WKc9"
            b"okHn4+WIEzZJHvGrh6ub9lxik2dK43d1cq1TZDqHZ5MsJM4fvTXzqV1iZecZm1fSm7IprZRNF4dYgmu/WFKDTQleC9r2US36v67e"
            b"B+axxwHQFvKnjPpC3BIn2AvHkbKSIYJ8XP8Y5yTS0hIumcOSWEtlgitZxwGkUIri+AotiDL0h1FH5VPmhi8c+18RLb2l+3SNaHqV"
            b"EnoykT/QHA5Y0M5KtnvBLQZrEZfN+nyMUtQiQGdUr0IlTeExhQLZU265wcxJZTk6w00bVulTrYObTH2+r/vbR2wDim5tgOoRK1l6"
            b"y8f1A1Z6Epu/bP2AbWN2to3eG9hoBSbtXQr/eRv1UTehMRosxfDcaHBNyPFvXPrvgybrW7lSnDDNqVZaC4g/DNGxtGXYcqLjbdwp"
            b"kwD/xoYEuwJutVecCGsJMVgxr363aOoM1P92bdV/A+q/htK/Cf1/jNbvr+Y1ANk7Qf2t1swM1H/DhiXWMAKpW2YsC4B/ATC1j+RS"
            b"kNZN3fYRNGB/6KH/E7C/rCH9cs/CrXv33vrtHvx5Yr2hLjaeSWSCxyoQbCXGPKlVU8xoUl4F77mTxCQkk5faEkUciZ9ugrELmP5B"
            b"mm5CmvrheGVq0vpO55greKbrEKXs8xfDxG5jvSun0pFhdopMtwFV192DVz2Bg1ICr4P63wdtmotUPwl6+oM2vWwExgVhgnBQkiom"
            b"GYBlFoIxCCglQSgepAxKsiCIp04yDloyhECTEEtY+jcCe/kIrLcTHEqV3rHde9O8LL81SQuxAlhKh1AasEM56EujvXfO4EpI8bBa"
            b"xADyzodHm09qkwN4pit7zz5lbrBa5PdwJcWHG1m+o/ZzQeZDZZ8CD+xTZvT+q7JTfwOwJwZgPC72lSdJro8Rb5H01nuskTVUm2BD"
            b"MAIQ9UZxghkYweMHMTtYWRAK3JcGYGxTbj9Z1M/LRHhgucquWK4OxU962ajuQE8buH2pRBe6VfopBzopnSpecalCa9+xdhkluO9S"
            b"LJ1avJCjVounxWC0uYlQtEQeHbDyqqrZ+Yko6vE6wLLL9BP2Iny26xyVVVQpkO2u0e59+MIcXq4iQQuBiVWCU7x+YXtd+XIHxQC4"
            b"+vSI9QOCW2fshRpHsCxjwuXDtIU3N5S0yxJJYkVSd3iGh1IOvYQ2ev75HOSNzI6SLBy+RX+tUGC0xYfR0hCuQLO8IsfSuhfcPChG"
            b"ug03C/aGxTZMO2wuzYOKWPV/9q40vXUWWW+of1DMLOb+YNz/Ei6DBkAgS47tk+TL091p+7gky4pTFFXv0IRUI7lx/JdWAfl/BzzL"
            b"/9ViTOX5VlxDX1yzK8V1zaHacn3XEd7EgqrVgdblON1Wh9Jvk/0akf7W5ce4Vtyj4Ji1CjmrWLKORdQqYgmyDGkcNEgttOLMIask"
            b"0fEflNDBe+K0xfYdaIkvOnBfnMKNEgiaJJArKwxAaTLjhRJw6BB0OenEuaXbwuPCH0gxLCZMPPZWaS/4onnNYXGq8WCXFqd0AB4w"
            b"ysqVpp+VGxedX/HpclauiN5Z1Ojhw6aocj0Mg9h6IQ1crj37Eph+sr0vPT+gBJbLbxeCmw2S8rW4vgycTf7OF4I8zqP1bK+2957r"
            b"JOBBQ3lXRlh7KfWjXUHhuIR8/75KMwycpP5tGPgw9b9V++ty6hcYCSksUBSwBOutiDtZnjS/jBaeMOqsIABcIC5iLE7O4paroIxE"
            b"KtjfZJVww9zgtQq0r/QCu2x7kEJi4pO4y4F5yTq4M6S4y/YIl50fvrFBQr7Xt5LvJE3ODGrltCsjF7Na2DByuErHDwZ69WxvNUiQ"
            b"fwYJbwBbCCo814F4yYXXhHpuhcYKERUkjZvBAEHTrKUYAAtJiOAotb4dMKaU+a2WiZOSeinhHzty3yqpj2SyG2aIF0tquF4rw8Va"
            b"GY52hg8q3/46Tnozx+hrhTg8sXXIb19Oy6lgYuctykkBXsKg2CRu8XxdpFoB9RSRfpFPGiTekhGD0wIaTgpoOCmgz80b8MxhfEob"
            b"2dHNFaD5MwX0n1PioHpWxAUrFeaagMZGOGpihcIl8GAZMtYASKmZ0shwqaiwIr5kktyZsJj/+Ts87e9wzYPheg3+790abpS3p2Y4"
            b"uK/z52Y4A9XegxnOTV+x/4oXzl/BfL9gtoJQKbAnQfqEbcNIci2ZjkWEVMyImDCdkFTSEJLcI8fYSCJE6kVwav7gbzfhb2QBLxT+"
            b"2sZeY8d34kV4In4Zyrv1scdyMhWTvAXVLJ2y++i0oxJFTd7rcXUFloJaaQm8hXaXGj9RDqQIaAVoW/7kDn0iKMJ/70e/MUxT9r6l"
            b"CUFwLQpRPzuVCy9Eub2UbZ/BCYK4U4ao4MRjct0fMu46Mk5YjzGjAYgDrykh2kqkpAFvSMA+xK8iiymSx3/inAvqtA0h4zQc9s8h"
            b"476xx3ZJcI0rdF3vsKV0W6RiF0qAbKnUt4BgxTdmxQGwNY4eWpSpkxmj1OLvAmvwAIIGOUrJ/TrJqvU62vBfgf+RVb8QCmvvrIWQ"
            b"3jjHUSSXvLTyPpqpXHo1IYFlJRXbBaTjntf0ulUBjg22zyrAmYUBTA22m8qu2mof//XPYPuawbaEmKWYjQUaYt4YgYFa4ZDWklvh"
            b"HSfExmSmAiXOKSlM3PUi7CVoqpUi5jtJFq6o1nmae63c91q0sQLAh4dQ1pHe9mX2WV9DNp/25ub9HlWsrj7LZ92pvwM09BAZjCr2"
            b"Fxu1cE8JwEdFbhCL8A1b8Wl8Q2gd7wtfzkuwkuu2WbVLAByuKJ07HxDTKLmilZbjysd+1oBWEik5umeieFJOsv+tM/kG7rX7xNSN"
            b"UdJTjOkSsTZK62eda0xNDK7/vbGiqaBk7Nul6dtq3Ce0tHfqMpL8mIxLUcmYcHFfrrzHFjgV2iumrec4cMUxw8EzLIR2Vsb6k3BK"
            b"aCwzqE4QY6Dho6zhhv3LHhOAD4L/lx0SnkvDI67wIMGKdRretSBXGwUMfN8vLQ1ANpDCrdYSua8mBzfBifjDpuWFOxRECks/WSNk"
            b"MwyfaEXggwyELMtBytOqYvQOf03lBuS7xbYFbrsXYi2AR8tIuui89BCiGhqdqEZ1/Q0ifHmftC2nO1tw62xsb0k6AZ9yACOg+ODd"
            b"emfLfFEleBtp3WQf5y/TKwpsWk2paDOxklW5LbvSezeGlEeTyG5CNZ5vbaiEZoTVt2i/O+24/AU/6rve8Bh7FyH58ihLcuaUNzgg"
            b"S6W2ELjxnhhFvBJGB5XUeYQUXjtDjbUm6UIZLiy1Mq4E/kFDYtKOUGeN2fGqMBeSEHMhCXgkJHF/KamS79DDlRTng8XhhpWkUCqM"
            b"FqmcKxzZ9FLV0H+mT6Cz7HlZv5uslry48BtW4vMqVtYtT7IEptVzK2yLj023kOWILP8rpGxDW0xzvDc5ZG245h1K+TW266zIINW4"
            b"3tKd3NgTcqC83ij13hlcpcrgNZ2LUu/uLdT9SWFA1MBb6EEEtAYR0AcKvccU+20SZ0mQgw7GiTT5o0S4lrSoSnJQFcWz0rjWZ9jL"
            b"Xi0x8kxZaoyN2zbudDBKYwlxlcYWYQQxuWEGQSb7W5L4c6A8oooIQfjvcSC7Co+9XjtfbSncM+u6D8/tB90demkoN8ZbxbGnDNtu"
            b"+qEdrdHPFHlzVGp8XPUhWxC8mZqMCkYLL80POXFRK4GLsA7eDilTwb24Hbdy8hGrxI5QzecfG07ksM3D7ClrsvgbeY2XxKwuZqtF"
            b"Oq890jvP3YG8eu0d0T6qNdEePfxR3hO/zotMGmUMT/guKaUgkjigPBhJbFwpiGRCBgVJsQeo1sq4+CXm2CBkk/Kl0j+pQ/KcLuWw"
            b"n3G1PXJjXbmmJnaxL3IzS99pYAwyuqySep/RaQUVxrshplqZz/2Fl0yZrh9jtF85mbeKSmS6+Fi+MF43pumkii+abOmC4mqgtn2c"
            b"qjrmo1ZKCi3vwTGlovn0MFLOKGFlVXuyKUJfIiX0ev+0IcDsIhDtVe6Tf92QZzK+QoBZMCzI+P4iMG5IfM48sl4xjnygDDSVBlxi"
            b"fEgeKEc2ICuxZYGg3+5b9AZPth5VVrVfH2mefd3faIwqm1m0fSfDoj5maFj0Z8n2HZLk832PlyLPlPAGGRcgEGUVA0Tj90VjYrz3"
            b"IEAbqYAEEpNZsCqhOpxnAlGHAokLNfl1yLMNIUFWhMQIVraXlCeDtsT1jrlkPZksFdw6NTvGX8KAsVXabVNCU18RQcunyTI4UsG5"
            b"plnbmibtvR0VxWJ0h3l5HSM6FMLsLy9dFUb0WfwZuaOcM6v4zl6ZVXxnr4ykEPpHf9Cza9AzpWVcqZ3g3CBNiGPUBIEFZSTJmUka"
            b"UEpXIv4tJLQC8oLyuChKQ2L55hT/UPraMJVd+tplv+aVV+K1Xq28HgBn+z3rsHbLlghErp9zHMZyCFDScI1gIrReGMBL6SVq5cgD"
            b"nzZHYaR2f7F9dt6RUlNQzpkEcMt5Go/bKSmRBHOy3t8aPHDW9xDrbrUDA2fYMMGH0x3cg+NbpritDiuF4WVI1y0w14RHxaYo2rNa"
            b"a5bGTjuVL/J0eFyHbUQCTLcO15LIxGIREu/zcQK1iifPNMP3qmxch6GXJzJNWHwPxIzCOIH6laFSU4a4gICpBgjBeCwUE0FKYk3c"
            b"QEnrvEkeOtLRX2bW0BuvTswa1KPN4J9Zww2zhsMm/Z+bNbTXMzNruOui82fZ8J0sep/sCT607n2Ta4NWVoqACSdeKE9D0Im+iq0J"
            b"ClnElHY6+BBzDHAa99Eo7pmNCSgY4T2j6qM4gS96TMKZx+Sw33Rt0v7UUOmqHDq+NlIn/xvKl+cAgg8lZY8u4tWMhuEGkDpo1ZWg"
            b"RTpcoiqcbeo17Z4cLZFHPv9+3Kjuh7Xy774uW9VeqP/xJ2vEWVALplvOmsOyMjIHVv+qFhhsGxtDMqrgSdoCu5WPcVXY1o/LK7Sy"
            b"gGjFC0dF7kNZw+7ROo+RW9NR7hWw/CmshKl8+Vc4Ce9KuloGITwVzHNElDLaxy18AEa1s8QCsThZnAnMhWcoSOUk1sx5Sa2wIH6r"
            b"uNZlGcDrKlgD1gGuOnTDUXJmFKQzE0rHqlkHj8or8uybnMGpOcUTaKlVKoUIflgM8BDBmpE/i+9vK391AiZIdyP9lDONrcOvabmm"
            b"9FPUXIuzayvB5X04FXjL67J9x+FllgMo/oT8Vv763fOunIuOl3Uzf1tXgGz1LNPEdqwW74Fb84kTHjLQesvKw8MKv7WxGNifENc/"
            b"E+LS2klGCAmOe+m5o54HSUIsz8EGbCQlgVJPOdYUSRxrF0ykiUsIxFpdcvg9YN4X+gq/ARf8UvxW29ze8h6dALJuGCDfpRVnWNWy"
            b"hsqKSkHXCzoes6CrEtFu9WzqILnDKyvBCTBG4t9bB+MaDgVLYPE95pgeOcnj6yux5fOk/A/9F77pi+cA+jR4N7sc33S82GEHsmZU"
            b"NLQJWUEQZoCEs1d2ufJ9WjfbH4yleK9KnP8hd9+0LlhOAbS3HIIJmIXUZwfBWLKykBYB5yRYwWOu89QpzJPThUMUeUmMIb/RBmmQ"
            b"/tVK+VLPmVMcZRa3T3fZmuNI5xqvFd2nGYF4Kxumg41G1UJfRHlXjbjDFax9bpTpEGrnTYg9fQ5oaAf1czVfOYYC6DtY5LiOlZsM"
            b"6bcr/rfAZNble3rD863Jb8HiiiM3NgesN2l7w+PB5Yj0hgzTdftU4HubpAUMfnv5jpVDXmGHdGcbsTTq9yTOa/AZr9Qs2mcr94PT"
            b"ivzBmw0Gb3YYfN9i7Mb3uQO1/Rs/PBzpWez/fg3Q9ueF9M4VQ3vvA5dOGBe3CcxTij33zDJsHXLGYOBMS+AqFi2ACHISeUsNop74"
            b"8A29kKb1Pn+qB3XH+G7Q0+nWvX2tKXtu0uJv1455Z6eXDXxy7aGQbFUg08SwDU4haV0EAbxxP14OoE/vE+ZaGdNtxepARzpxfLn/"
            b"LWyLqlzWVYJX+bmj9n17RIorCwpHpNqIVVuD7vPme1KijyoW4w1F/rCbigWCVRtpvlMrYYR+zRDpnrTc3KV0BlpeadydP0cFnRlx"
            b"/2CbCdSP5oJzdHAM1CTvPyOkf2KEZBAYK5iLJQtYLSnCinBPESEUOxvzOwREJKNKSoads8iEIDmABxGUQV/sIM3bRx9BQb+wa3To"
            b"ygxBwS20GE3QxXwlcQBtaW4zZfbLTk+X+lD3FoMGSLlPw4cfv5wwp3dCeDPuHebcdKn5JhCFVAO4IW3np9eUT+HlPVJDhzXq72Tt"
            b"/gxaPwvc88nJLsSvCRN3snXR96gQM7wWMMZrbqycOfjOtDsZDJx7c4yVNMYu1nvGXiv139XN+Uc9mzX7UuSYINYyjl1ARiLkCVFM"
            b"eqx0kEKgYKxHgVkZVMDMABcMqMVCUT1nWM/Vh0504U/r81friD47MN4q63pSOtcfKg0LuYCaTwSISlJAjwWI6EmzfSb8fkGAiMxb"
            b"0iMBogmJZCRAtGxEHwgQLbepo/+MFIgAzcWH7ppm/HbtoU/PPacclqUQHZHwHs0xXyo+ZKgIyVmTGh/znfegmA4GG42E0AQ74aXQ"
            b"QhuvY3HJnYy5UPpgUXJpZgp+lKLERTWGl6tF3BFseKHsxZlOkXxSPHM0kGyoJyPFn1y+XhLxTFewttEVpu3VzPDcJTRrZyK09CfW"
            b"xgFMNORKZP4U1zUzT7oNHZ1q1m24S4K+T745MgXPiDRnFJvZ2XaazTp3nP8bHTyqj/jTk/i0noRhxGrAzCEtYqo30jAsuXPGo/gt"
            b"FUFZjUFpTrFiDlvq47XGClenFoNy+j8smH9D676IUDQCZ91wVGSpireIUAybIem9sjPRIjgv9q4qPCI8zfTVZDXC5A3WhI3F1daW"
            b"MC4dYbGO7GCoKS1YiSSswpewGgvaXXiKS+sTE/BQOSMHlQsHJrkYjA+a25xjWC4/n+xEpHLh7vAQqtlhLwxXzwbrV0ZwlGUs2FlB"
            b"44GOBfSow+rhHXzhn/z9ffl7o5hAoIHj+F3xIgCLXzsrLJUMDNXaciAaOwOYW4oQ5RQ0A8HBWqWJ+61Q81ea0l9Ho9/G6X0Au15C"
            b"FzV62rwPX+EXx5YtWaPjz4IIF6OjBuUuXQ5JVsm4t/YkR3LVNa/miWEqObqWztxNj6EpqFxlLGIwG3mLHoCCOI/+noSUc3rLz5lW"
            b"sG96ySqUVjU3HajDjcxK19r78OjQJ3lgOvrt2ij/Gfi40ZJZRSAEr5yLxblgzkGaBnpmEwMJB61UIDzEXBaXAaOU5SIuAMLIQNkv"
            b"4+KLR+3cL5HsZ9R+8U24+DdZ9teo+zGipF/5mIU5MILeqaCshn/0Xady8xUZEEPHCgE5OF1UwpXD4L2OHP4YVuynG0b+jRQuAEl5"
            b"K5GfUfHhlIoPp1T8c+/Txu60GwQ2JqijJ99Njf8CJbRO6id1+jvbKac52noVZPwpDfYSsbhJRUIzrbhP5meKSZAqVumMS5/IoSzQ"
            b"wGwwXMew+Nf0BoBGXRC9W+dp1nymve5Qn1xR1Uno8mvO1hKviOCqN7z8ttcEmqdu6X0hnhwNSJdFkQOfAJW3ueL4YvJ5098tLGX5"
            b"JpZZqdo3b3BJhr+cLv2UtYY9OT9zvuyDbMr84svaF3+Itt1PuvVRLFcCjNKatTm4oqYJkqKzbkrPyh8sA0dW/k3lqfQluy0Byms4"
            b"Hb+kpT+jZM4weAfO5UQdv+JyfqzjfUOU6lLHu/w5zTvel8Sq3obVsIETayTCQVqu41KuhEq+1dIihwAZ7VXyh0XYp2pBC+ooE5oC"
            b"Q8oT6352Iv7KoJA+MHSiozccIOWaa36AlJsJ1I3QatNLuDiavYiqI1lw5UZ+veVf+AgqN1Xsy1A5kZSThRq+yQAwLzYS55PZdrFu"
            b"vcF9PEPKyVOknDxFyskTpFwjRbX1M5qHnVZVoT2+Ain3l33b7OsYjV81rx2RxkughhhNiceW0xAU1hbHEIYJwVRII6j3kjEqfAg6"
            b"FsXqHYzGtSOxEY3GHYlzruMDouPXWY4jkuGoaXGd4UgPF3Yg/41huaui80L83ivGw/Vdbdom5ag1R3K5cw9Jxe/rb+Khb9z9ereO"
            b"NF+Kecyw3FRa+7Z5o7kiSL5oIukiTrsk+lqCvz8oxZa3SH5Q5HEzuYRB5Zv9DMGQkPjnQW/IRq/NgEbQr+0p8KapwOuuQilpaVXe"
            b"1ji74p695N/62c4gbB6Wcx/tVKsexdbG+PZ0wirXLljNUfO4kEfIR0iDp/0Ipyk1lnmtqbWCUue0g/SAaBpYAB+zMrfChqQ+rZOK"
            b"FVjLHA5EYmLcf8wipNLZL1i0/T1v7OcvwuXi++QZVsHfsTbu8M4pLGVttuWcLXTQcH6L/0iOKlS/1op1Kch7JDYjJZIu+qvFLSpd"
            b"bauWyPMXCRX5azEKSCd61izkidSZu4r77Kx99qDRSxaoHF9fhZqFd97oXergkpLWnLg8K1Vqn1H/7ESu2Yk453mi2LoQuPNMSSMF"
            b"4sjG7T7E7BgUj99CRZnkwTLuuUAxMzIbDyDeGfJMGlwVG87Mo2/hLa5A284L2KeI2fdE8Tb6dM+dPmTWI3/hCxTwcT18HU0gtjyo"
            b"9ktsmp3qjGPCVprJ8fYNJFnripcfq3PCN2VWKuUgLQ9Ha+lTlgOA4BboPD4gf9YSG0tTqqoMfeTplAhCuyr2Jjf6LoUE1/wP3JJB"
            b"ZkC3hSYCdYcAKi5dRYqWLUP6SBWZz9SqKvZCLv72jOjT4vUC8/lNoDfncQCczK4DAm5sYFwJApbS4GKxKmKaJip+p5wNQlsVCND4"
            b"MpMqeKsR/u/x7WjZvlZSE0ckVvrXlGgBEd6LUjTJuNghpewkaRs40rGO93Pfdg+xZrQIXGzdUTEiu+V0mf4G8EFKgi3Zsn1rhcvq"
            b"hOhEe6JpkMSoFAkINT0CPsqOS1hcbXAFxW6Wvfzae+h296h2vadn97h6hh/6iJ45jE4wZJVb6B+97jK9zgXuNcUWgpRBs8AAKHis"
            b"k240AQ5WxuITeQE+EATS8aCED1J57LU3b8H0flGaH59J8/MhgOppRf0hMncsz/xIkpNVLOGn9PkfFHmHLkPR82+KzeF56VYJklo3"
            b"js5OnMNKAUkV7ogOvfhZDsn6RATttneDnFxez+JEw9LyQlG5MirJlbLyXQWlnDIdzl8Zj/oXi5QaltWcbTxyaqU2B1qcPwFaO9Xo"
            b"L4l1BMh6J4D2tAHqgTuivQ1IMsaCV8ZISSnmFgsWUJA8/ok7rxWRwZqYWeNzq6mNsQSL8Nmd/4PR1TITxnMw7UtlOi/vs8ekNnjQ"
            b"ISBth6BntVU7XgJPKIOmy8mstkq2eL2B7Rnb9WATqxzeE1bN+SWqNgfbMlJ2T80V58hS2W4YiS0eKmW15pYVCWWyEtzYIo+xJdLu"
            b"vqTAslqRAg3YDlk4dFWhs1/Y4nvaLAGLVs/sqHz+csCXZloM03Qf72TvMetNTpsBZyLM52sBdNLKVzP6D5bD/OoE601NAM+Yj8la"
            b"g3fEMWsg1swIAIIjRAFiTHqnEQHsMiEieajSEGtnj42n2P9ak5XrdLVLHdxL9KzbVLajA9RwFMaWRu1FNlfiKcgqdr3mcb/jFlnv"
            b"C9y7vCYxIAqPj2qSaInLa9l3pOul27tx5p7hwOVPdpcFN1auXKdkzcjscRd4orE/U04bP4Ts7XJo+f5vzXCPzGT/uHBv4sJ5GZQn"
            b"Arx0QRrQiRWNEOYUxfyvHdVcMm6QtJhzl7T1mSbMouzLFReSD8F7e1Gic0WiS6hb+R49oAmAt5O6fC3G9jEcuBeKTIFZj60SV1sW"
            b"q6Euz0VVpyewu+kCKKKsXrBm6kA5jtAxZrdvko/QuvcAD0/V1a2dSGszAltW3nsfc0KFnBIqngXoLti0XyYI9I9kf9b8aULA8YsS"
            b"v/bAJbXaaJOQEE4LrTwojbTESAfskbCKOiJQkBpYrKgVYPStcGFiDg2DR+iwLyrFDbuaV9LyAQ4hDoivzd6v4gUUg7lBI/Vymr8I"
            b"TrtMlD6xnF0XkFuus0szpxfnGAI09n3y0U9RbVa1jNeHdAa3famcoynOFDUY3qRjIz2HZvzFkxk73ed7FoRjzYjCc6NVVu7lf45K"
            b"8+yk9m6sBys2W6f7c5QA+jn4tGnK/hK5+F2tEKuVS4JrQnkmtGNCIoSUC4GR5EELQI1nGJwCR4VRRHmZ+tqpY2Kx+YV6nP9IPfP1"
            b"MqCkiFQe+GZLusczEeNrlfbIf+pMceFoXivnybacNLeGMG4L+dkasJhWZbAaascGtbMg7ryaxGKUy4GygRfhicFupszly6PxrwKN"
            b"D97u80EZKr1fOfAfq3W+WsfzsQrnWdw1H6g/tc43tUB8rHAE8cRbFywBHGTMX2AoQQgj46zTMaVwAyApJkQl5p1QHDmtkVX4G3pA"
            b"fUD07Xn7wKsuUXfAwe+xDzzPrrkveADYjHJ4jTQ54pRLZHGV3dAp6lT+pwSuGnVYtseMW94lcGl5bzg7VX2OAVo5f+xywEfNnzi9"
            b"J0oxK8ln2vkZJlTNMHH3yt5pwWNJisocfBdj7qSahzbifM/1Pxvq/IPNnwKmXATJMfJSKhMIKPBUOMY1d8bj+BU2CBksGHJgwStK"
            b"EUdpBTAWSfatSCxTdOE5feXBgtIl167H0vNNDqCULpOTBYvM8RYyZnqniEWLh1fDU/EsK1wUHTZFlpDNBXt8yqmTyYEOsxK6Lywi"
            b"6c1zJEWwae3DNoY6WFpBUVFAq3YcXc1qO55Njqgh0bcN9jgGeivPntH52Ern2wl71TN2SudjJ8psrFZaY7VMG6v50OwluJGPZNQG"
            b"21fyI/4IReTQBkk8D810LGclJ0Y6wR1IIZwEEWuS+NcgguMKYl5EWgjKkSeSM0U94yYg+CWzv2sNks83PWadjLaNcXv0eN0N70br"
            b"4vN9iHxIlknjBFj/mTd0SacgESMbsbSb7eXv2nuAJx79DQFfNwQMwmAmEkwOPIeUJzUzWBsZH4NFnknNLRXSeicCToKW0mlirCRU"
            b"B23f00/+CubuhfYdVxsPh/xzueWbPmMh4tGBWmWHeU5SYQkvJvDGW1ukFA7Qshy0ZjXCUY32GiDR+JL8MQCubuLSvelh3J2M8GUR"
            b"9+GQMb9l+s1gTNFecTT0vebtU1i5qzvOrhvo9Tcj34ESfcS+7cc+lJYffoAzafmhwvJRWv62VcgtXuBcDX6mLb8MAOth4ENt+R0P"
            b"t7MBd8foliE4VJa/hKX7iX6lD/rIn3cyHbUVlLKKKu+c8EqTJINBgvMyOZdKwQxRIJn0AYymCSZCfMAaaxQ0ZZyIPyjICRTkCs7j"
            b"iqbQDTRGn8zXMuaQzNOlLOsPDDAVTexNufkSVvB0qAbqwURaaLISVMyenrmSTpyDY0FNUQcGXElp1QE5KksOM4JkY+k3iE63pAT2"
            b"2hgDzMiBxHhXoShVC880NKBpaMDWlKC5S1vxZHjV7KhUNNpnrCi57fARUuNHmnZGBQCZtzmKYlGLE/nDg7wSDxIM9bGMiHlZCAEU"
            b"wEimNMJAlRIiEJCIBOKEk4pTLgmzTBtPlNfaBhUe6GOQsUKGPNXImAlkfIE/c06eGZFNFqoJ7qrNpXitGC6d+UX6wuPcyEMt9qJ3"
            b"gV3wa6nFsQbvkQ9A2nKGT6havnXzYSX7dT2BTsJz+YuAQ2Q6Y4LBIVj6IRtVhh72N6yExVWBlIJmFX7vuUeXSPv5NDngOQWMVCLz"
            b"FxlO40OFW8lg0j1Z1s8W/SFOa51ivh/bqrV1z76xX9JUF2NE676oigFd8bkSs8/LV7wlO6iaEk5RLS0OAsffvyeY4sCUINw5zjgL"
            b"1mUVYZSVguP2zkkZMx+xSMZy1dq/cvSsHL3Y6k25pdhuMN51sfeFb/PdYIUwJ4mkh9KtfL+a+NwgKT4ddD89q8kT7RHp7yqHxp9r"
            b"/2PvFAziL3mskvRHATugYL1dXWQMKW8LIDgZoRy2SchNe1a5OHJAoZy0Im/NGdO5chjlGLWX0L95ucoc95F2Ma4awXjQFB6ZZzxj"
            b"LD1FK3T/tvuU/vR28bcsOJ33ChGCwHDwAWwsOTmzMTdjyblUhtkgDPGYuVhfKhwocUpTSgQOmPwMNWF06sLBzlw45CMPOTTI2cWB"
            b"CK9Zexw21fNl/ezwTM/3QAbJcY2+W71HP5IqSmojTPVwrzEmIgcSwWSlwLG16HtpthSW8naNN2B1fKesBgltAKyxYWIP3PnW6UCL"
            b"XGApZFMXvmeNcZtstwhS0lqdspWkqBTbmydQK1c2T+YMaVnvzxt6XQVCkI9hXd/EAaNRE2Z7R+uK08UrAQhBpWaoN96HwHkgOr4d"
            b"OAvaCKw1kU4FTrAXEGIFK42O/yjjZl3HlKm1e4e7RWup+X4hoJFPxR0w7TOSFEPnhzNJChjq6/4DSQo69KyYsY6HmNZHwhS7Ws8d"
            b"YYp8VFdrPhCm2PWH7ghTjHG6LxOmuKs99CdP8TM0iSq07rLZuoDWfavbxtWxmkDMxlJYOBFSn8IGZ2K57By12MT9JOLUc8IhSC3T"
            b"9RnLQMTVQSBvMTaEvqM5e0PA+K0SR9f6h1s7o1LyHPRit100asTYOrvM/LpM/yc70WN8u1fb+iKtrkj95efXMxxNbnJsM9WhHJPE"
            b"3gCTVu1tLAOUwtI1Elk6GLvU21ifKH3qZf2uB3JjWEoOybRuwveNAZ8pPOWohc69a46OLyRfb4l7VvGTJHjLPUBbAeXuXdv1CSul"
            b"d6pklmy5PmarzRytbeZaw1Bep3W+szDkiuytHlYMiuphNUXbZmU/tE38RfXP1zSQ88UtqXhNwkI7DMYYIDFLYJdoEzQETYhmEltB"
            b"TbCaK+YDAsOsDrEUUQppZoEzSq5V6if0h0+Mva6C12YSby9i1vV/6gcvutmAbFaQX6rGeyOPW2JvDwp33m4zFq0ispXEcmacl987"
            b"RyZbZFRdBj65DTk0d4MYR6gvhXlXBJegJ7PovdGaPKl/ty7FLgzUPKE5W5ZEuT8uGkGy4r7JirmGF92ffO5eI7luYazVr/wQi+LT"
            b"Qmz/SG5tzZ3GS85ViPkREeGRN9w74TDRLuZML6VBXiFFvbWGGmakNQKwEMgJjLm7777Bpo3eeZeXvd5841kQcXJBTHMnue5/5eKn"
            b"3Jtv4iz+fuaCAfl1gjZ2wTBqR+OKEQw3HZ5ADRvNgLVylKhVAs6Jny/oK7YREUaGRilqNdw4InbREz50Aq35VQg+gfV2VLYUSIRc"
            b"q34yD053IEeuPd6bmNt8/Tc6vBOc7GakuZWb62M5Rd0uQmu1zloj1Aa7gFovpfYCxNVL8bRzO465R9wjfOwr7TgEoJTGMBc6/jcm"
            b"tvi10YLK1GyKKY+5uHH3xnkFJiVFAkHFPb42IK3lDOCj6jpVipsOsvBLnTHHW/aOHHUZp3pkr93U9nls0zFQbnygVXbQLm5gBOQK"
            b"kIDkwmb/OEcwQbH+2CTaVcvjQE/gCLIkfbxLnDWOH7i+W10KXyxFOuuPpe5AD7Teb8IJOL1VaI7hBOevjOEE56/gQ+5cOQn9o6ra"
            b"/AEI1np3XhebX4ITvMebQwAIZ4NW8UsGsdZUWpBghaTc4rg315gHrqlRzMYNPKKKAo9fWm18EBQHaX6ctPvrhNgv7vnpMpNLP+WR"
            b"80/Ic3Q0gvc8WgBJyz51POVJJyyRV/hogJa9/nKP1vsAbaN641fkU6afFYlhduoclc4MFMjWlW3PPjwwh5e7GMtqLPa5E66JZ+0d"
            b"PZAx8JxZV2yVc08XywKvWL5GWzjp+RIlkrDFzO7sUspFl9BnFd1vd15pjWCgra7CXD+YnXQUEtKLV4SG9tmsz1rp3xw4Cb1g5Sz+"
            b"T+L90xLvAjACxJTFKhAHBAeH4heeqWCJNIF5KhlnxDlmFSMS+XilynHDhFTAhfqNq8PV5utXW8IH148tNRHcdan3r822OMil90uw"
            b"WtziOhvPLpldVEEbSDucKC3cW6OKnkNZoxhF8rhGHc+vismdpLesQpbg9FM2OmsnB2WZinxZwEGIHU0nNkG38YGAljdKymvL2/F6"
            b"RZ0emN+nHPXkekFeJGkMlYwxdJLGuNKZrxXPzrDH+FQLbSQ8sWOKV0369dHfqvD5VYHGbQNQEv/nvGGUSDA4IKsxjllfC2rSRNoq"
            b"IZhyELjzXGsfNxnUJYuQn6t6OakmxXpKfEqnrTOwOLIGbiD3RhoMR68j/IRy5pUF8HyT0usQbQvA0rHiu4cePrw9XT5Q+slnqbkj"
            b"mpS9zUagJp0sfXdJctnZ8PXjqcm2plxCCmQYah0NNbyM9P4lkBEMG3dvJTcfl7sSVm7MlwQvy6bgVoOHN6NE3swS04N9J9A+kxnL"
            b"vSV53jqLzDJ+0xI/YqH3zk73aG+uv9jj70/98tnEz7TTxhimsI11IHPMYyssp9wyaxkByoxzMTtJzgzEOEaxdVghwyXC5hsm/qsi"
            b"xS9wBBSHGd45mqGbbBZL6VQwbB853Y0dnHZ4hxy5LBPX/e2urAC7EsVqBTJD4F1dd0Z6RROgd9oxLTPHTT5iglNGKyKawNLPWQ8Y"
            b"fvTyeTJ+ZZn7wilksISVk39U4/gJBaIjoQ+mvfmzV2DwbK5AdK42NH/0l97/SXpXjmpkJMYebNyyag5ex8QXAnYoOIycpphi4mLW"
            b"B0c0Bm8EAwRcOvD8l01fLzhxD+eqZzNDOKUm4mVDeRgtypTHusEiWyb2d+fDB0Lektt4q1mR3nAz2QbccQ0HakCTSS3MiZZyoVoO"
            b"P3Z516SaEXMyrjTnN5F6cdSmY5v9NladkKg8TlBSUKZI0pqDvrzBckXdITRjp6XqfabWtfd9jfzNgq/hFNZN9xXJUj9mq7gQwWs5"
            b"Xz1juXjPEkPlwOoJ24rzhtA9kE1emzdXxZQ/NKNduy9VAkfL3+AXHEfEAhEYTGrfO6bVXlEUAuJYi4C19FIkO24ca3IERgROsY9f"
            b"Ku0sR4jRYIkPQTga0wfH5Eewvl+HlOlVOLoirBaV3xnfdABq2TTlK/8/Pgq9IkvaS8qz3aJp9OYXYXtHRfkTUbihojxbCMH9EjAS"
            b"lC9/brTjgWc5+dVYtQ8odqp8Z3ffdG36j2nNfxNNjDkg8J3KFz3NW4CNf1VMYu4lcBZkrEk1VtY4ZbnkHhPvGNOCkfjtSpgU5QBJ"
            b"op32SBtpP6Qzv8tWLL+CeRqjZ+IVj1OAWPDMzR/qXK3+ZoqayMDTVq4BF3H5lMZapd3RR1iogUiJtlHQqwstWkfr2K+yfe6ljpDM"
            b"f8N47T6LuWpRDlp0jgg0ss1krdpw64ZdAuPPtUKm6YB8KxYZzJ7VmD5a7vdIzkbq9W10Ckq3jiFERm9wFHfKkeWC9qLznkJG+g7d"
            b"LDYrB2qo5n0zN+pFN4j/r9FtezwlrPh31Vhw/HCf/H3GjfoFYhmNslD5vj0Son8oovEmqglY64JQToGmnFjjqSIuaC2wNgEZgpkH"
            b"bi0xEFMsFTbw+F+NAkYEa/SOTLs4+MyLyDmv5Fyc/oRR0k2W8pgLsy51QcclySNBxGllqzSQs80D5Nz1rf2Th43FobEIOcAlGr5y"
            b"M5w6z+n7Pai4zAlbAgqqsViJPV1r6Oqr154u7ZJJ2kVhUS81ZekU99akfJKcX59jjtwej80ZInSxfluxdLjGx0320XJJplAlU2gE"
            b"hHp2XfVwKyO/uyL7VLbynerqhxSGiZKBBS2TTKUQzBkuIClVMs8xjUsntg5Rw5ygDoRhPNaOwQuEBfPSotcyjfvB1PlU6p86cF6h"
            b"Jd9j9U7GOd0s58hPxrtgZVcJl9ZhsXerrnF+xNBB4ngNp2M4eZRluEAq7o7KDOeFWLwgA5bysHyC2XElevEp2hnGt2dF8ffPxD2m"
            b"cW4TbviA+pncGpGduFkt3rtSjJsni+ya3BXTn2cc1/iAXzYq+ncDoTWJUiJAKBsoE4rFGk+mHzQmTQBkjbWeyBB3YTo4YrRyNChl"
            b"OAhquadSvGPG/8WiD7fzjQkS+AviOPtsGy6Zv3Vpeu5WLPpWX6+zAxOpnTEdbzTCXshwj2vSHFIEb5hoHNcaGeBGaaYw8w4KORXq"
            b"9UzwGFdjnqOOUJE8JowPnJKP+jis8PMafZzxZXxdH+cZ1O2RdszW4Xo1Xq/HNrCNe+pXVqH1WnQdn+Jreyv5NRP/GHztG7Rw3jbG"
            b"wZQKn9R6AZxXQnKjqOGeIysYN54bsIZZr+OmmmGtbXwFY6UtCYhwa54RJGMne2+YCjp8xfjtXLBhIp4gq7ftzSDY+uoFL4gcK15r"
            b"BbGcc+oEkcISiBIhvIVDhbDqc26Kyyja1Q+iOuJJmweQm80Dquw3oKfq5dfjD9i/AR2nEcoy88xW+76JztRjfWYXwVa7iBqvWs1v"
            b"cFOPts+6GU37pLVa/xGiDU9vul/pFSGwVcpTxySVYGLmgoCJjU9SCUlwvAJkJWGUMW59fMLSSEcKjtOoWll47ab7JZXiSZl4UzLr"
            b"4sa6s5FRy2/qgZAX6e1o0NKJXIYoYtVUgKXvXFWcqESt/QW8SMX2eJ2830dZ8hPwmRpWytUphCwwf5yB/rvJDWt0xHLU3hxcPYR7"
            b"GvbjyvnQG7xXqN0WDn9GS2vpBm4U3PrZuWz4uCd4rz34jWu0D1Zia6py1llmAtE85ichASNnkgaHDxoRKwQjzsfyzGghMNEavDeB"
            b"C/BaC+3851PVuY44Om0qzhwQW37MdFCQt1P7W2x2hTFAsvVMbN12lSwjzzfHR9T3YOfbX3ttZd52/siAanmpWBLLTDjZGq2cpD03"
            b"q6OhAyuh44y19uO+nLfuNvbSe9ybakA11YBtqgGLvtUuwFqnrnqogf9XGxruA93JTGP8cBnofvu23YkY4Fubc4esRTBzzAgev9eK"
            b"O2AWdKCIOIstWKyVibvKuJjHAouYGALCJksYAGyRFzT8XLblCyccN3g5N8gtu+0BMH7syB35OKfjjoNG4cxdkB82tXLNaLDq9+91"
            b"JB7blJX5DNr1XdSoBiufLN8zPgSgHz5juoL0CUmNt8FVKbpsn9vD0pWU96BKkQ2ZWAvBHo4pkfl9WMzPfP9dbYaJR5OE8lst8V+i"
            b"4Nx1RpzbEMwlWmChV27VYmd6MJNogQ0q0z9q/RBbj/CKg7PAa3bezh8f51/wcYhgVsb6E9nAYlJHyfk7gA2AeUDM+MABFOUBK+01"
            b"OOJN4uN4qgxhXJNvmPkvpPSrTMt3ZP5LGii48uFGa/0odlY76Uy4UUn8Vwnwt5aU+ehouKKcabccbGvSBeSEjDEazFiOrj4lMl1M"
            b"Ei3hI5/GgTkO4+WKkgUNrWg4m4PlkTqffkUl/NNce0mk5EjeyvszP+/zzH8GU1/9Z45DeXyS6cf2NePYCrJePflbBf7NKsA5jTvG"
            b"oADroLTCAiUfcWkU4UoFRBCxWCnl45+EQnEf4LXhWrMA8bv6AxUah2P1R9q03dj8n+t6na07XRKctjgOc6wDxOnKevVoSeGPlHzP"
            b"JL4eLhNHt5qspFVWUIJag2BZGxc3opEoUztlVt+ibAo/wMO2UN5cpMuLy41CqwRXJ8A1PTq/YTnySRmum145M//dM2femZvvmc/v"
            b"cfY/Ut6axdWP/jS4Pq7BRZRTRhjFlQcmQMhAEBc+rg0O4m6AUWccSoYROG0ZmDHOUqIZt8gRYfmvXRVenMbvqfemmJSiSGvOu7H1"
            b"u2hVFiXg6uCIhvrgFJIhT2LpxNTaJ/woI3wdqwtL2C3Rk0yhJaQCjfEzt7QSW64dK4ZIJ95by9j0+r1v1b0scWXB44jgsbJOq/ub"
            b"r79EP70g0HtjR7kl7vrxmTPFnHVwZiS0dovqR2OWwq7P3h4DNSj4b2H4BwuDJRxrqh13lBoBlDtngMo0wmJSWQTCU+Qhi/YGwZD1"
            b"HmHhJCeMwKcYs72h+ym1/0re7lROrvDxZ1lSjHwrplxcNsO01cxYeKR1BQfz4zGB4Rh3NFDbFGnPaA5QsWa7W3yUGoNlN7L21Dsf"
            b"tCI0hqhormAYnqOW9SZ+4aAAVXrm9LZCsgUt8Zztxa00C1P929UbuMFl1I32gfAKTMlisCFsYeM6zPo0R2/hb68b0DBeZ/m0Zby+"
            b"0frilOkQs6CM7ykCFSapB8TEGBQzLLiglCdUSYys5cELFzBHscz2ICzyOCAi1RPmauIZZ+CR03yNHYE1Tw3Fr17qQb9B3hJideDX"
            b"vua+wmcfuK2jpWUj15gCkWUXlRnHXXsiVzgHoAX3VmLlWiU+MolTywbhQOztTOLgEMQXsS/Bqxu7yQE30lHJZT2GSYmqsejsAuMn"
            b"kMUsbR0Ab9KNsoHSrXjep13Y70oDzJzV2JQ3u4Jyd2VYXh01m4tu+NxOi6rTA/gqkuSl1uo1+2DTTDlSESrt50tW6S91WKMUe0WD"
            b"sDjEjQwGFtMeDxw7QpDWEqsA2lhrFVaCUMSD0cCcAqEUtpa8wxJ95of+CePeKqWRJamNM9rWluVDisL+8gWOwhr8UpLCdgUzlsIJ"
            b"omTAZ0hqIxs/QVZQ3q5BwkpYzNbbHE4OJ6EXiQzZ/awQGT6B6J0xEOSUgSBXBgKnNQOB71rX5UZXDAROe7Zs/sM+PDtstH+oE/kH"
            b"/ca3rCY9Z1pxR6gjjuKYyoxPFpFCGwRaCRSznAJLlOKGGdASwGgZsBOI8DfwqubaeDOlgG0v/EjpTj5jC1lJiszViYcEU7KY1vUE"
            b"zPR/sh3yk7Mt+HC7DJt6HdkHQcerz6/nvqjcLMNmO/VeGIUufw9qV2ZpCZ8prhBbV1pU6wh5aB2nz720IMh6pxpp1K5diorrJakg"
            b"dBvQ+EhV5Ys6YUNVVaPR1IiqenOLfF+IdBW9W/9YeS02mh/LCkos92KuwrzxptArZR+va0Bei5AuynnVw7HHWKUldcEz7MP75bIv"
            b"fpBAX2QW+ZrUmi9u6TCuaVZLbZWkjnODrJDWeGyRlMYYIuNTjpUCTTSXhCf/L6kCts5Li4kVLPyHzSJnbLFuB82W1uBBteTYRfx3"
            b"eIWb46UimpKTJeFNh3A4KjoVZDlCn9MvcMU34MUVYPPBoaMjSEGo4c0d4JItQrn6cgBPftQ9HW7gPxB/jSWUM4bZKvwiNumXg8lb"
            b"DoMv4AyUSHfi3mRpnHrLbInTarjEaTVdonVV2zw5s4ccWbl0g6XKHrIJKf9+Ev83a/r4rIkaa0LqIgQdpHQJnxYYcxpjj6gPgB0J"
            b"oB2ysRgXHAfkhdKYSo4EVR5+3HrwNZuwoensuU3YQGf/ep6+aBMGT6Asvm4TBtdtwmCUjDFejL9iWq4ox+Wajx/smrEYHGyNcf5J"
            b"DjCHwQEvdBbrDZPJEvkl2+A/h7C/5eDNy4GPf2IxQbn4BXfBIuaNw4opyzSl3DCEwGtqkY6bBcql1zz1op0HB0IZ+BXSYRf3A08B"
            b"kocAtYEtDbmG6yVToa8iLQD46BPfrnJb1mb14jLV4MJrY0Oi4VrUNmXQEgmMI/S42r68wyrnSz/ZyuZeERMDc/scBjke2KleRYlI"
            b"v9AndcPuEbpxNXzDDfQLV91q3HSucQUEwx0o7DQzd49G4K/v3L3+BcphDEutNXbSIGsVeBsraqmND1h7QJISaSE4boNKAjtEEBNC"
            b"zA0kWXoB+x2yjM+xByfp8Fp2rbh++DF3YqrTc0jFQ8kf3isMXfNkz9cmj83ojaN9kPtde+70AX0xh2Q2Nya0rcsraaFBvzv7dlUW"
            b"Chufb3ZYfoNySMqjdP/Iw1xb3Ng50Gdz7V369qZs28jcriqLY+ADnJpvwYloOGx1b03L3mrc5uFmvQU/pvL9UXlXG8okkYFrSRLp"
            b"QhiO4/6NG4mTPyJhMTUbBVYRwyiNaTf+G2cSpQRD7C9TbKyUhBYdITmSEroqxLNVpPVQb+KI8BqpyPxeNHeXVwlEdt0Ld4zDwGXn"
            b"LxeLWbbG0QOwImX9GLVSE6ByM8T9JDYFqXViuMyNh6H5oyj+tGrjXR2hpwAT51yEg/jtGevgapH5J9NYIySY5ThgQBawRtoRJ8EI"
            b"ZOJDRJgUjDANkilB4+NgQKMQgqWCGZsUwcN7vF2/0sOFBz1c9rUe7gHHPx7rVYmHt1E3Oqc9MeHiZG/cCh6aBN4b2fGdMoyB7xXD"
            b"0pc+ohZKXBnFEUqrI1jHvmq1H+gy69vncNVR0zlcQtKuczhcmzDUbznBtwyZF/g6+eIYSvFyHfHn3omu4ulwKsirbu5bE/aM1XDW"
            b"DaAVu7h+LKfw33GfoOrq1g3erXvQPvqmiX3GdLjfvp1by77TJudy+5ajmBWNR5xwD8YDi+WujguHYF4wCdbalEJQXEksJR55RhkX"
            b"yvIYnXx0PsSHoGedCHnaicAvHgceiARycWfsdXd6IgE6BOH8OkHLB6lnTKiDy+U+buOnMAhsxTzJKubJDk2GmZ/X4cPyEoXRDmIj"
            b"++hsOXOPn0GFHyGADw/iB9aH2MaGgm7NjDa6G4+luLJYCinbI1plunjTcshzyGK40y2YKTXAVIYTln4uVP3cVpLzmI1hy7TjR6Vz"
            b"8AMwxMu+f2Q2+xhF/EJuBAcSv3rWWceMYoY67ZhDQVkQ3msSMx0xYGywwGKZHLh2Pq7m2imLPVf8m/iHzUR1X6eIc3CgHgoH86dw"
            b"aHxlcwFt5WRmplw5MLsNYtFpjk029UeMAx6JE19tv94qrdPp8oVSRFldLKpZIyDFbYRZqnb7sBmIu4QR+kXbsNvA4HmLlK0IgYoT"
            b"QWtThimcYAMOQw0cvmweWwXcMY/9Mw57hk7LCQQvgXgrgiIYZGCSSWlB25hFY05NiuqKKFBCcKcpxpjrBAVTgVmqfiM4+A3Y29dr"
            b"VN7UB3ssCTzoPVwEguWQNERb1pvVrHtlYrTB+bPl0IsgsDw1KqAuuvV/j55kB4rv28B85R7mYBaXMTqSzGw/c77uEvskGEy+qI6d"
            b"vTKvY+evbLv/5uG8pj0++qBY8R/ya9A64EQYldJ9LJmlCtwFlVoHiiPMhPGJAyKUVcSjEAgKHqhFSlGvBNdzNbKvdZfJaWH9LN3u"
            b"hQI0FyVlZjkf9VzlgWn4qDw86DvKKiOzo/1GD83dy9Ujma6IO2Z0Lllcxpb5f0dOpq28o4JN3lGoLQ2ilXE8peIN5NlQxVp8LM+G"
            b"1v0Feyg8BjVn77Hw2K5kORIeK1dRolu55O1DD7Uo490p4Vv6v8n4ix/sriTOcT4H00bwebtiDDKDvWafLACztA9DtfvfJZLzoHX8"
            b"efmcUf6XEpCSMeuT4BQKWOAQ639QPtb4EDcDHrQPNkjOBOXKOe6FlEGEdJRQn+Nff4dx42REd9FzuMdBzGCwLQj2LqiBbRgx0VpU"
            b"lj3YQbah2m8slpkrUpddAMqxc53jfNLs/44p2x05cTtlHXEDU3hG5MYVibcHNjoZ/FQqA1qpjD44n1tmKsrav6/ulhih6sQSvLVo"
            b"bo4Ay6p3p5AfE7PLK7JqNbePx9xAOOEG1tC1toDv/UheK27277AcD8jan0F5DMjaPJbcxDtjYtltKbdcYI/AxL9TjIVKFnjKxafa"
            b"aqFi0lY6+XUjApwjp7k+zckwHuitPYazpDxu4LBvJXRWtYjrBjF7FRb5aDm1+/Ot7nzwUH9SrWlP3UAs935Ni9gEx9uH3Y2pBqJs"
            b"J1hjWuE2uncZcbpVBY7uE34xxxOyEnpv/UCaXL2clyxLz/oxatMpfHScyq5+Ko0DJTl0gmb2Vjm4fPi2u35TdW3xEbmlu8Hzqrm3"
            b"yUktvrbS3hqW9fIaWSTXdi+R+ikrkOSt5d4+62ySq4Z68++bE2kjvvEtRdpmzfa63dI029vei9i1uS+Jtz0aRKKuFK9HmSQ/JtUS"
            b"gKthpWfMxjKbehnTukWeG60DoUANdoJb7DQE4YLzOnDOsZWac8UDpUFizj7acqnwGWVHNU3jdN6qedSn+ZIm0tIxeeQyQlaNt+Nb"
            b"Xi/0h0S8keRwQZcw0fZ9OkOt5F4T36DaKtWfpIte2yyEVF2WusGOOs50YkzzwwJ+yO49ERCq9Nv/arK4cCED0nIHVHPug8JcnrDS"
            b"4hdS9cjLH2yzmKVPVYKe1kPC8jYH79jRPgM2zxw55Bq10UF2i9ZzDl6Hoqvivnsj5A3aR28jggjkBHDtrQrgVCyoqSaKMw9KSIId"
            b"xLwL1AYJ1oCSmiFGEQhng1BCB/VMOV0S8etw1V+EyT3bAnkh5Plykm1S+1hD4lB0y4El9kW88AFzw0+AdPN0OUZX9zyZcrlTS6Ze"
            b"/67VljhEZ/W7dHZZYHFtMp4clGKXJM5x0X2vV8fh5y6ROf0/2/WIX5CkoXyTtgc1JgU6972RI2uGmUAtVgdNd4PWxnu9K1/nqrqp"
            b"1jUPK/7eBXe97w54/kL3423lsWAx+SoZAhfBx+ziCHhMjRQYSew84gLF3R/XwTlBkMUoOCJC0AwCUlT+pInkDR7IhVR8db55bdp4"
            b"Wj4/FDIdz0NP6CkTUc17Th9TP6Rh+OU1olztwndZF6/twhszjiOl+ui4xB/ImkKWq8PFWRXTQZ099vModBycvFVhYTS273bU9yNq"
            b"CX52JsngpgTpmIO4SolCbai6NUjwgXe4t04ORXhngrfm85H+/Bp5FCTt7fT+5pKfnksKTiwCbCwzzirriFNU2JjzkeBxETCcCsUk"
            b"ocCRZMRxAc5yhYyiXisUPsTeJi9zRzrys0XbdLkhBp19cf9XLHrWtkJ1usv+e6PVg626zxc1mg90FnXgVvcrzeDuDCgsQ5jFJeXq"
            b"dJYsvy9w9wl6PaX0GVUlaX8zOZJ79OwxLu/slXlHYv7KSI2tf/QztZc/qLC8JilNA8VWxKJUiiBpzEgGuVgu6MAJ8kliwgUn4kVg"
            b"oJ6nV5FC3DvGhLXwWtrJS0R9TmDUD8Um7ytNnggvtnVajBCL9FhtntlMhbJWu6StxSavPtXynrQMztRAv7Edr6UIQGirpOQKTu7r"
            b"KL7gjBFfStQtFXYYacRLUMygaO8e5fN2CpY5Iv6Axpqj8yUSUJL2c7y4m1aWM/7bsv3eSrf9cSnJZCWTI/9Hq1f2cq8VmZjKR1S6"
            b"kj9SKeeDejhrgjKOhlhDYWIlAguMec2BBqesQZ4C4zRmJZDCi9T89FIFT+LmmknPWGDvYHYcNMQeEeT6Oe4tEYhH4u6bCgQ/hE3I"
            b"ZS+k352xR8SBivw1GYha4JEvC+m5DMRBb/jIJxnIQFyRY5/KQNRveVkGYqyxP5SBGIYusg5JB4LFj7TT9B5LgpQDaCfve5ew919X"
            b"hfiRpL0KX0BrfAH9h3S+wTZaAtMsbqSpF4T5oOMumjhhHXYx3cd6NPnPG2KEMSRWUMElwgcg521AFlH+Bk9h+jnNs6trxRcch8fG"
            b"GRdnYhOlIvEPHIfJUZntq+4fUAVe9Ck+8e/IkUdB4V5dMoG8ikrwk5OqP+fhb67H849Ud9Z8SpWwnDIhqAYSIO7tg+HEamGZpM4E"
            b"BjGnYqTiV014JoICxDwijGslKPutvhlfKYovaaMNKdU3DYeOWQd3amVzMMIDV42b68N0TjWWdcdbPX+jziZ5ijLZ2XyxuM7XLTJ9"
            b"e6GZLBUzrial44+eD0iWqgRA7SDcx9uGHL9sG57jTxeL0TvjqjNDpEX+gu8ogvYpK34aGbiwGWpsz9h05WgGVTMIbzfAGkVv6IRq"
            b"tPXHr/44v1oyG7+5BknJLQZtNbfxTzkQrkO8JBM33vExEOOx94oHgmNRrjlnWEklpHi+Rbwg+k8Au4+oF+e8i8eLRN9SvUqEmCbT"
            b"maPQrE7diArsKqPhoFA0XjGGJz6tU7tu9VP2hPVv9Nw545ZtxsEzYzHCEGTke9eu1GU/8SVCxS15tjMe8rj5Aevwq3qGKyY07kQu"
            b"DqzmE4Izf72Ixb8lTFSNjkXC5A5d4l0lt/RgeMwHWqBghaXEENBCgAIc/+OxEDIkUXcivfOc0Jg2pSQadEi4sc+y4b6A38WnKfmp"
            b"ZvmlNHMkzPVyagcZoF0DqINfZV83RGEAdm1DU0zCRSHBhg5tsHzdm6Mmgj5i/ej4mWI2hcjRdHFgCxqDis22FPwwjGxVIs4FhUqv"
            b"EPeWf2UNRZTg/oj0S2qkn9IVlEhgBMmBFV17vwVbAr+mLZe+B68QoaBVr5l2IhSy6lXLBvnQWNtVT5p83LdFOljvPQDvt+9Nn3Lc"
            b"LvSg3wTilSowwjkwEB7bNICk8e2xcohRaakVDnFNjMAiQDCGI2ctkQZpKmJhTP4EOV8gyHlRce4b6Xb+TkHO0ne4MeKTa4t4N2qr"
            b"0K+0SqjdiI/XGZVXZxhja+XeUpjN9qqAyjDuT4fzPZWuCkp647TylgbGpAuEUWWCDI7EEhgcIQhL4AGsEB4SfoN54WOgptZ8Q7TG"
            b"5b/8q/nxWuYbiYtVUgqEPhQXq5QE+mhVsj1whTbV8qUAhD44haTlAcQiudlVjB/rIQ9E0XCNBWkvZSSKdtTDbI9IcWUd4aiuX+v2"
            b"evt58z0p0UfGwhQ7wnbGAoLHKbyEEfo1iEa8FXf6xrMalk3rXja1WmIndW8tjlzJJO8d4a5PTAfHQN06/kNp/BuUhlVeM8uMQI4i"
            b"gYRkTiFEjGHccQtYGYNxXAIQ2IAQ95YjGwQ3NsZj9dpq+d488XyY+EzRuqngrJxZvGa29w0dp3DBRxaheMjlQEvzYbmu9ZzHhmsK"
            b"of8TcutSrKFdv7jcjRQHKygY5zZuI5dDTp1KayG0BhC95u1l1lr7mo6gG5k6nHaZsCs9jy8hX2qOe9LrM/2GX9EcnmMx5q4eM1Rz"
            b"02uocBdV26F6WKExloe/bAb3jyZta+J0PuD4nsRSxMB46wh1PMR/B6GJNvEhZUJrhDQgYpOPXvJWphaox4j9psT51EhpMNS5NFIq"
            b"6e0Arh0ga1u4GZsluAzcKon/YoK7kY4vp8Kry8V/IGWezdNmKROaNEkfztPmKfMXi8L/44QZtGYBW++xlsI5Yl1MiVw7FwiRjlrm"
            b"PLKYeewoJTI5yjGR2CIx/3DlPqqFQy7w3EYJ+IViZCN5R3QiVnNCxN12trtoV9OxyCCBVtscDxM0LuJeeGjh0XRfWN5yHalze3wv"
            b"bpP+SvBBA/2h3vFELk2tfvUVyhdXZvKkm60VoBvqKHwHUbES8m85uU8zb3tYwlD9a8Sq+PYSBEsLbGAF91hQ4Pkp1GHypHAS7pLx"
            b"bVxMcCGApwG71DZNIAEiRVK9EByApdxHk+KiYRgw4mkkZX8u5e0Gm+yVHLYb+/WLzeC3je4XSpwi+NiKvSJ829AZriHFamxxr0ib"
            b"LyTHX25/pnuc+8SXmr7/rl36GsvNWSo9e+XM+AhOTInaf6ODRx+05vxrkw7bpIqZoEB4YZDlzEKwSGrFAncGkIxbfKIAc5FKVUFk"
            b"iF9/S2ONqyR4GYT7nFcFeYNU2F1d3JOp9uYQQTaD5ws+SEMtlXySLFmIRGUZNDQLOiiK8etGTeMKM79rrjquitEMJdbJsHrNMSWt"
            b"UyyrYntqWYFlqaKpwucyNSVksbfg6IIKTYl7Vhs338t7Ug+jfbyc6nfJ6WDrHM2AlxJ3xxZ0vOID9uCb+QTdEqj5okDu2/wmVAJm"
            b"WUGVxmCCFwLFyjgoqoLV8Rl2SnMPWGuvRSytrbVJIBcb50wwTv55AF1x97lkEvQiD6D7QrVoyc1y+wgnOrgpLLVfBfrf6h7ckMqO"
            b"R9w0JeKLiwQ9uo72HzUFLXq2a6mOW1VJmJTs6fLLMZ+0/8nmzLdz8dH/R079f+RUH7dk6dr8h1ZZWp7JPXRSPJxWZhV/7j+vzcaG"
            b"OWRi5vVKU++BhIDjPxodv3fIgDECcSmZoQwU08FSyZCS1AisHfDwmxhjH6VJXZtp3SOnPZiA0cOlHqda260faMsWuxZMuGxFdFRF"
            b"Z5ZPMfDWARcuvRKxJtPxafMV5NiHzLeBQsMzrLG/Mdcfbex80qWcoKkD7IKzzpBY0mqwnIsgtVVUeIwtk1oa7jwHwRXwmFPjq5Zi"
            b"jz392co3LzRuOHAgxBCk0BAX8urZXPOUalAW2plQzLIBL2XpTkQVXwKYXfJkbmtcvr/xCcrgLnkh3QWikGqWHNJyL3osQQpPF0WR"
            b"UNWbkEo5oXurHLnRHp6qWu8rJ8jCP9jcFupnrLATNi2E9tmuq0AbWYVKdWFnMDRPGoWESkShethJKNDK9exPJ+el2TcmUsMpAWW9"
            b"YBi0IEZYHBxyjtJgPQoMYeQM4kxybgxIqbRjmBtk8VOt2m2xHGXlScn7qXq339fj3gC+FLFradjXr5tK5f7ybHa21I8pbZVgOS8h"
            b"+1ViXAuKgmZVJAetJ50WgykuVaUI4T2+xlz1Q60UmCnGsFSx1SGPJSTGQhOwBMQ1YatG15M2RX8OqPVz7xaiaU52b9o17qDC/6Dy"
            b"NmgfnyVTOE2mUGvC9E92NYNvZ/VY79E3LEGX6Wqa86VS86XS3xpAegbE4VgpUOWQ0F4kUIEKPlabDHlsFVeWSxco8rEOFUxIL5QS"
            b"hhjzjgQ3ByV8VjPsYquTF5NEgY9NvhasIMpUay2qxNz3MCvB5AKTiEYKZvmuoJOcSx8ash+NYVh5I0IktNbplQNX53Mo8ySVEAVD"
            b"8avBMXMz4qMcWkaGCX6Em3TAMFHACgLRCberMdxFeztZKTwWB+4+poIl+EkY600EQdlv16outdpLiw14jBMYS+L2ruhwIg+zb+4f"
            b"zas+jWG93yItuIBRi/QRivVtLVItAwclFSOYESnAU+SIJo47TAwzmijrghBEam8c5jzJxDAv4ybfIuPcD5Y1/zfi49eEBHA1XUfl"
            b"E6xCOgOG7orGAsYfm52XsHyhfIh7PVxyvoaiCobxejHbpRxFbsqZs9Aibqb9y/lnsulYrrLprNNCmIqms81LjB7eaYj7KoGQXMEo"
            b"pRXMonFW7zrPXC3h8Y8EyGojsSJqjweUsHK7vsqVvZG753Iv80pZdkLl7Ssjr7BD57Xu0o4wYcMS+aNErj8M2BADpjVwiyiSghHi"
            b"Y4XhGWeUcKdxWgkMRYQyzHGsPsARRgUWzBmAYHUMgve4Q9LvAU24jgp+gGFoJf/aqvjlauknppATsMJhnIdGSInxRG9o8DjXRccb"
            b"mK/TZR8YZtTL0dB8t4TmtTnuHBUeH9WY/KRLKrEMg2DN5ZGq+oZuhlj02sqSJGo8Xmc0VNrCKRAI4WiwZ4JmIUrXUCK3NeIpBMS9"
            b"XnIlu9s8Ka/xmtrLaaW1QGq0Q/Vks5vktSYOP3aRq4cZq73q6W6GwGOryco+8qGuwk/sLT8wkPx813m0UIS4I0BGxoXCBxOSGBlW"
            b"2KngkyCZUARxYwN30nitHRiBeUCUUUextjh8K9nIMz83er4NOevbnFLdjo7qnTQZqdTJREt6i2Fy05rdY+EQe1CfhDPI8DJAHV1+"
            b"0uhKk0KRoSA7tVmsnca7gm1t1xsq/YbhKePHTaEx71aQhmGXZrA/m8WRrYn9hNp5ep/7mjW0KsRplQ9pB+fdNWsqj/ZLajaVJE03"
            b"jasS6o/QSSjF8ojw9pje+0LCm6GOGGRikcsRsdQhQ4lm2nDCLHeccq8QkUwpoqmLCY9jjSCmxbjhxdq5f9yJ/mLf5FqveUszD3hn"
            b"RdwVEKqmhou8ZhWWXpfp/3r5r8c43QN5bszEOOJOSSH3FordqVlEismEjJUSwZa2h6xAtN0nz8FZHUyWvvR2VK3Fe2hlpDuw7EAI"
            b"Qu3NODZkUsgCV9tbOLPOSonKyb+mU4x7KvmqS9yXhBoxUiJtRO6gc4vM4j5so5st5glyl1dZtHmyYnd5Dd7lNUZ3Qd5WD7fU2Tys"
            b"hBtLX+tbWbA11eZkvndi83BX5vZtjWjDYnZNgDKu4v+Sz4MB5xRlNlAjhEDMpKpTeaYQsnFL7BASSAottOIMvwFkVrNkB+rjU9EE"
            b"/Eg3YUpW61rF4705OeszHJm9l/BrrcjWDqwl7blYjhns7dkcd8tWwUJ0m4J3wM6SjRIiD6dMV5XjKAK2cxfkYAhY/hYpIlD9UkbT"
            b"QpK3x408wpqN36ZYW1hdsrI0k1U+k1V/VzbmwATve/D6WYnbYAzdk4qMMH5Y9uqnGe9DPLG1VqxatGXbfUx3clfJR2v+LPvzIyPs"
            b"5UbBRungqPLcWmCGCOcF1p55EjxDiGAtpBPGG0Z9QEgSHF/FVlof6w3C4EsTNbYZP30BiiUeobEuzdsOSKIbmgoTIoJoIUdX9p5z"
            b"SYX++s6MfmFYV82lCY/XWi4gD/+LLy4cmFv0eMtycHqDsZIX2fYBI4DXA/7Do4T8PJ2hAzDknJzvaRqiyeWg2qVsDGZb4+NPTmut"
            b"rwYtcVD8KuHt2O0uIyLeh3veZnNV2XnyPesTQF3VNn2CMQJ3/HCjlx0e1l3WX8aWeDh8+wc8ikFP1TgJgK1L/juSxNWC8xAosVQF"
            b"xVj8a9TSKOtTH0JRTEPgXmPOETjiMAnvGb6RU9OHZ/UWZuO1vto7pqChTsKNMdzJPGz5KKKHFJzOosYSBBcq7Ebc7MwkYe1iSHnY"
            b"Ig4EI5aVLZXEFUpPnrJ8NzPLhYqx/86nXpblXqSfDSvl7J7Acknpp2jkdU4urQSX90k28lvfWzbvuB/fzj7TAclAngouml/EYK1P"
            b"Ien3ta0TN2Ugbpf1Z/6X8pSnIYv7Ja3NL2mnkXbk47XyDztYuWlnVLIQR5f5cUPk22unFRmJB9O3Gs5xMn17rLb2gZXCIjAomVkK"
            b"xEJIwmpOYx+S74+nzgUlnZWaglYKMe9M0j/3yAobMI9f89cK8zaKkfNJ2jkO+mSSNhujkV0z8sEsjVZvMZqiraRlutad4qFJMXk4"
            b"SeuvvZ2hVQLBy+lEO+9KcZhVF8j3vQQ74lCuNerZqhQRl8o1kfLtrPf2UfkcecF9aoDG6S0I8n22MF02dmv7A1ceaFAZ9hzMLPnJ"
            b"wyWBfvvp2Qhu/HZp3EPbw2LJbHwTq4IwWCdHXialcSC19NwFTBw1xDER/wtWYs+s1tqh4LACJX6tnftMKPwxFOyARl6pa5dtyG8Y"
            b"wB+dKRsWCTz2jtzoEfCcpuVlrcpzP8hqelz7QS6m7xtW7Mi6gNZUE21gscVDEl9pWhRziWLnzreOzcHQvf/o69vEn2LpdfCdvDc7"
            b"rLxHOeZJI/c7CGM55XPIqfiDnBpXnp1tJOK7QsLOhH3bY1/Q3vgzan+qYsUhLgNGY6acUt5pwlH8N80Vc4wQYYEFC0BDYDpgRY3z"
            b"JK4SknJKPOG/XRn9YOyWyXusGr6pPommV+PdXJENx4jWlwFGYLJ0vMx8uXJfxCLBuKbuLgFeEyaXW8Miv7dq8tXRkjPHxR+0k2cf"
            b"hue4tCgSpnpCyDHLH8eO9HCr84lyC1usFJP1LuSVI/UJoSXfldAvjQDTG94Bj+GKxYErwzNc5UjcGJ7NmgBsmpdriO2aT3vjM/rt"
            b"Z34PFCAfTv1eCB2zXDgjaExxgRLucSo6tOSexv9oHsthAlyCc2A5CdQGS2MtTK22jHBm9M+Gjl1DaF3ats4pugcm3SXA11W82lHN"
            b"dkagKzHyClLrDtrsnaiumUjukCKXL3YTyV29fd/t7Hu7RTBuBMyZbnOb9BnAdsZIHv/rmjs/5LXzH0SIWanAYwJO66CRTAo3wQVN"
            b"TKwh4x4I4hOPQev4GuZUOyO9jfUki5cHnqqPjsxeghO75a9zCVF1WYT3OImq5kr4mj5uhrr1vLh+DjbWVJdyy5nd4AjfG7wdhNJp"
            b"pZWuuhUgS9/ijX28zJyGH6SELZLmqnEbosd9+jH9zix5xsl3uVfoMPQ65t6bc6zSIbqeeWcEhl1ZpxHWqTLsiNywoCFWiFv7pHaE"
            b"7MwhW/Zwl4iX2O8+qHqDqvlr+rmjDb2MG3qCNSggiDmFEfFcKik8ckoExTSJF4SJ4J7EOEmRsYZgAMNNjP0R3pAvtMC92BqezZHO"
            b"5CC3AddjOUg4VT+8KERxj39caUx2/e7BB6tlIQ8zsqEu5FzpkbaKD0Olx0W2cVFw2IUo2ZkQZRZyEOh57R0hBL6lwCPL5J9Xw//t"
            b"GRRcAN939PWzgicoBIUVKlA/hQIk2ABo7bMmyXbqu3R7nXa17mPBxz9nyduKj1Zr45jVVse61vMALn7pUVDGBKQ8CjogGl9iiCKO"
            b"vfZIKG9Z4NR4gTD/YXq78jmLtjtmaUMxBnHaPNw+lnzs9/akHPCcdtFd3aG7e8eAHO666dZ2lcfWRx5vZUAyF3s7er3o4YhK5D/7"
            b"DcAsVhrz7ICDbW/TALoD4r2Fz5rhqGaO53KKVJBTx/MGlVUBscbci8oe4iGT90PaB48aso1OTsmyc52ctdkQf7Mgho3afEnvS7SO"
            b"cM1pgJhGleaKOMPi1xU8xUhrLHUI3FqjgGrLdEi250SC52AVDUTBj6hrr1WZY+DTcXP++Sr5FEo1rNquVczXC+AekntmocPV7r3O"
            b"xN6Y5itRbpTwmFiwvBwoW/vE+xG4/mUcwNDpkKVChe3gbaw2+uA5MDUjnqxpn4AMjHx/3+AI/MSjv8L1lfnUYkCKOiaCZlYErqU3"
            b"nFPvPI/XoSmLpWp8kVtClCdYGYSkookEZ5z5pH7Cx1ly1+S2B+rlD0hvQ5bWjSHaI13emyyyHFhJpoujanqnnzu0M5sy4tpKeEvG"
            b"eGQjKUt9LancwjGaG6vlS0jBRGG2Dr2Wy9g+wOCwFF1sJGrpc7JMD9PHF4cdDltXiS8Z93AMlN43o+T1RIzXtSyvYbP8kpACrakK"
            b"tOUtrA2D2p5yZFR5h2XwreXTHw7PPiSsPhieOUcYE6C9xkESSOIcCaSrvKIU65hVNMJGOxlTFKCYwZEiBDNsY+K2Wpgfx0p+rlCd"
            b"cF4vM5evcqGv1Z43lX3PRIP7K1gKVYVy/an2QlXsoCz2RUrxzVzdNy4eZOth76IGAvdLXf6ca2NBtNo8p/rHObr8DjhjhUzB9mHf"
            b"EcxdouTXSMjxr1Amcfk7nWNaZ9/qCax5uaafVV3leYJfesa8YZ9x2vSM6aBPzA8PN+mcTkJ47SD/Wvu2H0JIDiANJ8Q6RZgx1HDP"
            b"JVVeUiMFSrZClnjthULAceBxCQFBNaWOWhKTWfit/I07FIoFyCvarN4uEMVXAni7Q1n2e1VZnLWsGKY1wGzlHlzO+v2HOdO4QFf5"
            b"0AfmxCVt3nO+RYnLC3L8zM2l1fIWuG9GiXyLhIR6GtjcrIOiRPl5ndJRggs3Y6xD0d/ldDklmBFcGjrlwsQA8FFCyi/vSQ7HLegc"
            b"XhhxOx3jMQzuGWH5XQR+tUOeATo6FMcR0PEKS6U/IsdTawIBImQAJqyMGZ97JbDEnGnCBSgtEJGYUB83EQBxIcAMBeEZMxproOB+"
            b"kkjFhbnei3UsLmtCXJa7uOjHSSrI3mroseRN+Zjv3GHo+lV1sGVB68DvCszwTPBiqcGLsQgagPSGCxhG+dpbW5Edpzfp4ac7E7cM"
            b"CiqPELQiao73qISW9+CYUtF8GhgtkyUsf/JnVSji/p0SdFOb+Dkld4Lr5+2zVXSokZzj+6uVTvFYsrhTLz4+LGdbBYzWJ39aFJ/W"
            b"oggJmWootlrF99dWGxe/xgisBUWC4ia+QpUjGnlmuXdWOmOEQ4gjHKR5fmDKPmcw+sLO0cV20EzvTj5mR9NT7c0Tz3u+2vvlvI7W"
            b"/gwsmJd+XYmRvf3H8d3TVzedbtLEudnBaXh4sG1zsuF9LT+6ijTUBiQ5KDVuNqxf7cLSNXeKn+lXOvE3LfZmpnhz29IZOgWmBD/Y"
            b"Jp39Izp41NJZflkD5t+1WUrWlEhQ6y3H8Y9DxfRIpUGUSM0kRgED1tgyrYJH1FlqtHIivgbMaYkUAsl/rILPv1DTueiYKjYZS7pZ"
            b"2u393KMw0AWZoXyqhMMmsPRhaKW3M9IuypF5vcd8L9mHGj1Qgr6Bls9NXN0cIzcj8RV1M1mp+cj/1YZFIzWfmZRZ9fAhLeRPzadO"
            b"WUpaAcTFXT5yTmGf6jfHeNyueBKYF4h64Awzgh2TNFAWCz+wcd9EGVfkz3T+P2Q6f8lHfsonnNjY58D3e87n4vxONhv3O+W0dlvx"
            b"w7RCBtNKGodW0jj0gSz7UaPxz2b+ss28REEwZZkyGiwS2mKsjUKBcG+VVwRL672I+cwYh7FRBHmHghGYGe0x0h9iVXyxPINpecYG"
            b"tu2L4lepPvYZBcEt0SIFxB9ip1kM4gheVMEWKRixd77a8/ESgpHi666v/mLsaUWUmNV+Z6XM9qmlLc0KoLwvzdIpUmkGCta1V62x"
            b"XbF5qEfF8cZd4IDE43PWXWzhtzc8ni1fVIpb89hTdIe7RF5YOAq7pS+upj24mva0Koukllkk+ML06DjJ6R6+xGLiE+yGKeD2zRyG"
            b"Q3UGinCKnUA2pisrUVwFHQDDHvsggsUB4+CIUYo4ykB7pyzhjNCQuGRI/h7n3i9MbZ439704ZvmKoiLMFRVvTlgO6YxXto/98Caf"
            b"Ml9EvKltSoVdZuQgvVWC03swoKjR9lphsIOZ/wC4MBuS51uYvXjF3iksN/NoxDuBtom5PEMOy+eGpNPUDolG6pA5qnzWp717b+Zr"
            b"Wokt0EZ4YeYVcS7WMHtllQWru4bdBP8klVfiDC/QC/9z631iRpMWBy+cBKwoi+tA/NuhgmspqBAiOfQyHLSjnmskrJdBi6AQNkgp"
            b"w7gR6FupL4q5eg48Us95AgNQT+TVcMzbW/s2V/iVNWlqYNEvDOso8ii1K7b1BqoLXNecbhE56p2djemHkmcn0o1Hid6aVXwUbywn"
            b"zsFAGUWjOX2XglNUltRhBMmDpM5BzheWwEZVRw2F2L6qqZOWhFsDIZ7XtrVgbp+dWLavLm9rE6J9BiejfqgdIVowVv3vW2E+w+/+"
            b"3AH8l1R2npeNJPkxGUpIxnzABKeSWi6o4sIxIoyzygqMuEBcq0AFIQzHRO1Q8CEWRSHZAQFI6436r6ns/FP5nMsqFZeIHvlt8+Qc"
            b"i07pdox8vSfLU4IyNoAQ3gzPh2fPOjvpSmKByzpk7fCAHFdOnpIm1EP3nkmTR9ZPMo+foEQUHTLeqJLxfXbOdwGc9hmcevLAajwM"
            b"tfHwYyGdsZzZn5TO20bvYEiSjAzaMMxiglVGxqKYa+bTHN4pG5QiXHIkqONx4ZZEKUdQIFZy7fGn5lj0Q+l32qTFx03uoE9b/fV3"
            b"e/kUlBM3AYweZxgocfHHxsOaB/eJfihfsQmP7bJjD1eDsTxCmn+l69gEwuVGoDuaqaUPkELXwTyrZnYPbOKGQV8czt+yrZwBkdh0"
            b"mHWmJz6TZmB7A/j48FU2lC9NeyW9oTq9fXE0/9I5FlgJwALxAYw1HnkdAPtAKBeG6kB1kr5VOF6LickMK6QsTnI1Lv671O4XOkfO"
            b"XNA7qN82hb/qfPic1tdXTSuntZ2Y6O9+wOKyhOaPuPOz6qO2heRYd5YDkj0jqj5MTbgd+jmOAK8wUqgYo1OHoSkofgEksMmVDL8G"
            b"Kbxc/tNukYBvSZGNKVpnLjsz14hzzx58cNSp5Mlq/8iZz85FMMIfHP9NrV6rLChqFU1/1YFgzQSVQgQUmABKrFFMIsqJMMgrjGMg"
            b"aOSx85RJ6z46B3wJsOHReFB9NXtftaQYZu6hiuRAqv0g56g2kYu5wcU947Rh+pNoX1m43N9+Wy1gINg+1M7smbX5fFn6BzMsa53H"
            b"Zl3prTHSNcd1kTU+aVCxovuDZO4zsHyTe3A/uYbvvzmnI4QxmhjA9zTSZ2Qo2J5V1r+V7o5cgRkbjKJTSie4lkpfn000Fyra1GE2"
            b"9+3AFzf00muw2Ukn953juPO87C2xxAWJibFUIYFAxxxtOHjJA3JUWMqSeoJWiBqmOFUeC4k9Yd5g9cwIbuX8D/P1M+ILlYPFNF/j"
            b"U4ztc1ZCO2dg8+vpMLLTBvFxeJaztsSMt9ckd1WmNTWmmHRlkkg68ZRs4nM5nyEDQLfTL8jerUVCurI+h8afolIf4FX/hczVJBYt"
            b"CejkD8rJAARWu3Bkf1poz3ymQQFN3yWvfEkPavF2q3ZMzRnTuXIY5bizROrfvFxqjutHcQMk8Eushe4JVuKqesYDWcqx1fpclnIu"
            b"comHfpZz38uPCFbe8BWakbMaDeCanPU1X6G3zd1w4uFg4YV1nHtpiXEGlKI8VsrBoZjHncE8CAEWgvDJj914xjT2DLRQHwVMPEjW"
            b"Z5X1q8rqoV0lemhXiU7tKo/l77CY7q5FFr1JvALsRG0P2VlfFhNgNfSRbBR6Ykz8jSGy5LxNiGYYLUsuxWiPZiOVnnS+FBW/aOSg"
            b"h/PQKG/SKErnyiENcvg9bYlzAzV8YDU8z1vdEWgt2uyaRNg3QZUta19Lh7hYlL7OlVJi7gkOHhFlEMHEWeWINc4D9w6cIIwHYpMF"
            b"pSfIcywpEZzHL5PS3nL+OzgPHSV1xnmAi1wGuMZl6Mx0+cZlwO1A/SgNmWMekR4uNDfyKdLrRKoRpvUmhSKdJJ1Mlby+VoPX6BPt"
            b"qdIpsiTLc2krFfX3JE92ekP9mE2pDuyE6sCmVIeZeEn18Efvtj+4p17TlyKgLNHcKWM0xHwFSFquaQAiLY2pyjAhnaaeea4EkypY"
            b"z4PlAnNG9A8zwnkOYvQ1d5sL8Ky+Z7pSYWElw9Z3ayVmlo7kYSQzmMckFxxJGsjAbg/TqOqR3VrhopfM/7N3XdmN88h6Q/cBhYzl"
            b"IO5/CReBAZEiZdl/d4/nzHikVpEiZblQqPrCbd+c25izj/rr3LG9ebTNfZIV3/EMX5H52ZLMz84xUvNwIfO06/+xP8bw5qP0gf+I"
            b"JHBkU+EZ16CR4EAo8pQ4hQWPBaIiijAP3gWNLMeJHOtBingIB88IAAf4hmxKKgmhFWTgGi9wLVP3Zoa7OdOfVE3NNT/CMEwQqHvD"
            b"5JzMyJeYUbTbDqiJOeM9vcH77Lb7KIUrROn0YnMcSZxYoYaTD+hTcdo7vmmDSxL960n63LCktAaWVgmU1xmU1ymU7y8fqNLjGbtA"
            b"lTYiefOc2qno1WJ5/5pW3n80gj8SqdaSxz1z4AS04zJYJYI3xkoGKjCFghBOALWBI5DK+yApOGSd8SqIv85L4a6rwef9FK4Eq+EY"
            b"5TwQrIYdhvZQ8W8iZTXVbxl1Thr3rh4wgDYHy0xjEvzMdNWf1nhU5n/iUwuLVSrZc1WZ7dNIPxsqxNWnAts1pZ+idmK/urYSXN4n"
            b"YabY3iWW7TtOL7MckFS0qOCjztVm8tsewAtc9ktWCwzT9Ct4MiPiFQ+rfSYviQOy+Ck0dgr01oTpBGOdFK3GQ3Iue9UpyfypNjv/"
            b"Q04LEgcj4mpgQsDYUYY5VZwpJUNwsSZ3WCOqCXGMeia0C0Z6mVYTa4iAQNE7nAR2UZ6va/MPFeZnh2LH6bAFLFfug5ZbcNw50HYO"
            b"6ZIDajZbg5GDVbsqs+XWPFG85rHOa9YcpPj+wZJ1aMUjJgePWA5U4jxFKqvJMd5fldjxE1RFt4sPoV3BnE+XA/cO7MNymeBnPIO5"
            b"aNYVz2DdYy1RUKFjJzyDrqfQ/usfUwqXkvcF0UCeaiGvS96PEg1IoJT4oOLqCBhZT6XDASunrROWEeK1Fd4xi4THxCHHEDBGMVVU"
            b"BQniRZ6aMqfecHKkfw52qfqrzn/TclT83KbnpTgsFVteK3smf4ogiJ3kqG2JHYPbLsThmshfGwGLxW1sWi64UnIhldtLOXMP20IF"
            b"VSSATw8qCzFc7xamWwBxlNRCyv3cUzmZHECEQjtx66BttcO9+LHmqEY38Cmm6Lni8wgCgmVDFZZpDpZQ/3pyfrqvVKzTT6g7fxRA"
            b"1PROa83ArYjDdRGHbwOF9ok4qlIhVDP11WS9HkQdE3TKGTcWYt6z0gfktGSuwIQEdRYQp0QiJjWl2GAXv1HexRJOCa+FwUH+TtB/"
            b"J+h/3gS9tDsfbHhXI3S5HKHLixG6XJZ3K+XT6uHvCP3RCJ1qHcArRQUJVjumqXeapj/hWN/Fv1BPHGeCKWYlUGd4Up6nxnqVJurW"
            b"/oqKfGE0/03aIxXZ/XQMWV1qCkt/L1Pt/CkG4KZmSTrjPgGiTK5E9MfLz7OgdK9EIdUABGgreNIPvFJ4eS/ghB924aeedVdslyjg"
            b"b8uNkLjxEeIRelw2zqr9M7yrKJ19Q36mPr7ZZ29Uoe7p1jk8BEraZ00nsCNx0s6M+zTe/pUc+YbpEMOOBaUJs1px6hJaSSKZHfKU"
            b"MbEKkJgwTrxXIJLwCMYGfJLes9ZRT37FWeXMt/S2PupdxqLcJlAEy1J6bvdxnJT0enYlkjC06c2dFtkDy6UElUHX6YHacd7HUviu"
            b"XuzdGf0T2dSif7oBrAhvRFwmV5venN5n3ZdTJuHUlrRZkT0XQqtk+wCTqTZ7TfgsYYDeFmZ96G+y7c15vVHnNfmH15B3XvkAyMrJ"
            b"5PkQqDM26ehD56OTPnSm/fLoV5n1p5VZmUCEaOK8tyzx4EzAVBEF1GJlgUtGcGAUQBkZcOAAhiDjXIrxHOO/2Aeq97ebEbUbrCae"
            b"mTCN7nYT/6fEB8prBUZ7k7Uit7cER1Si9qXnNKBrV5ScE7O41YHjnHvPwQb1JORsUvCdQkQfOzXd9PxLbyYKC4j+X/UnDW1vOr/8"
            b"nqxUXimfQ+5l1S+QFeReVnlP9pD7w360flYGQlW1Wz35DOz+1/2pKmF5EnTQVmODjLeeYuMpYOQxNQ6kEM4RY72ghHnspDBGS68p"
            b"ocEIE/9C/4qmwc3Sc5pz8NWY+zLL3WxDPNP3nHl5rnbRGY5eYKOl7BUtcX0hHZrk/V45b95pwwzkARjMtNhu3xl/spNkegKIGrhR"
            b"isky0G+bfX4fkfyKLj6Xcdqo45VzCq5qSVzNkHBLJOcXBeKFkNPvxv+DG39ONIq7DqTAgRYcuYCUkMIZhmMRR5GXnAaFNbNUWmaC"
            b"ZkpzAkESLwj1X4KFLvb3XxXjuFGj3BRCvqnZcR9neRtDeleD+dTE36o3cpg2qZkMyaRVvBypz0xbYC/mJpc8oEjrVsDgMp1PmqGy"
            b"GLctZFErnrQwWVzYA8WyviXFoxmyswTmK2KYUrqjOFTFaIKplHQJj5sbIOfEUC0OKGHld9CCOh9O6LmAeK2cPppyrQTzj/06rzfs"
            b"+z6f4Pp5+0xeykGthlxVM7d6uHAFPEWeqid/sUxIBe3cMvsNaOcN+ZAf2OFz6lhM/RLHfM+NoiLmeYuCMlqLJDgdOFVW8fj3KgPi"
            b"IKSmGPP4iEpKJf6jvFdWxitoz+xLkdclMFR+XkqkFzOqGortmVaiGsMAcM+8u6RHJY/UB7OcqBMplKk+ibIh6abbyIFEMtYJloz1"
            b"fyeQgqqPdscTxLOk91YlSYo9d/NB87tj/6LhN5HPkSIOQGjJ/r9KIl8Dgx6pFtPzr6P8OYjzL+K+kgg7xx3772gKBkUfVxKJpasI"
            b"WUJEx+I1Dbm4VsgphgUzyIJXBAEYBZqF+CB4QywXxinDqOHuO4rbZ40B/qo38JgTdH8zv65V+T3B6/dsRV5VnUPtPRjy8cM2r7uC"
            b"hckeO+KnLPcXzZRZWZs/Mk7r5HrM0Pilripu6f6dvNSEyY+rkpTfJPPjSjN7LMv51jKOFfBO7tqMCFciAExtwU3t+70WKbSmJ9Ha"
            b"5AQ3+IX22W5JyGtPwuO1jE7gDc2JtwYp9Ghk0AsF05nrVG2WcieR/5WtjFEvr5ln/XyTY1btCkS95gRpaoRnVoCIRazTCryXPsmw"
            b"aImsDIYSHItdwNTjEDNcCEoyRP+KNvE9QNgHV4t7oLHPLhSPIWiPDasmfZ25+sCjxvfbRlVFKuBUH72QQ5hKkH7j6OwA256j+or5"
            b"RCtKQK07VWELaCspsAIafJ+gwG/r+HnrWAkBQTLlLDFBUSyk195KjwnFyMY/N6XAYMWRl0zrkMC6iAAxxhNGxWczad80vu4YfygL"
            b"vrKtZrM3/BoOt9snv06CQ/viCofbBa9xuPgwXLlnM/4GDhfPzn8bhzv0Cm7gcNlLHO7Tfq4kUnL0MJWuwbjsEozLXoBx2SUY9yq1"
            b"0uN12iXWHwLj/nTn9r/rzx651SHCLVLaclDSCIE9Vxgz6yy1DASj2AngxuGYVYPASBIar5A7Fjwyf0WV+ll6wy1o0p0q9RlTYZOb"
            b"R0q0ficEtwXiJuqPZEvyPRfyQ9MfldlMrQ9dy/O3wRTtgv7VBr+y/WvjE+QgB8afe8uBbv3c03qwPSbfWu4CSc5qWX+5/4HhbsUo"
            b"JwdgIPf4ffEYD8hnLbFvFqvpG/RQBGvqVnJVq65BsaWOpRW19UGt2mkL/taq35dPkxw+kph4qrwK1AkXwAQAD4EJpDlQbrG1zAkW"
            b"wLpY14I3RDqvKFHqr1O/uqUO1UO/ugt9pKNFt45x+inHti2hl6SBWm+qSQ8EVwCHrb+7g/u3GrTnDpTIkQ6QO6ptskIbVnX7jMSZ"
            b"pcbPAXY6QPwJp7LU6tQ5Kp050QEOHZT27NMDc3j5FAlitUEr3nUjzgPbz3VgnOBDHOZ0GLrgkvDderGE01dcksOxZbyUcukl9Esy"
            b"V89FDtegh+uewxZYMYF5pd+y1s5ip0bV8HBL7lVDuAmpUBDz+F+tq/9C60ogI6jnFFEhKSUUB8INYtYo6wXH3HlOiMfWac85dzho"
            b"6wzVlsa6PC4iP+W/zb+2olwvJzP864Z+7aCvo2pClZyHOni051Y7I6ILvrDnPg9pILi7PfeWbHa4L10c0O4r5OzWJxbd87huuZ3O"
            b"1jr9Llxd1qTvksKU2D5RuS8Aq5vPd53CG32Yh7n2f9WI+6OJsrb5a/RhqqJ4S4j4dtr7qESWwB47GmImY14GIhgLnhlspKM6YCq1"
            b"cDHtJTWF+BI3LiDkjPKSIkZjqfzXoSLujp3uIx7udi4eQQfgRDIU4by1eeqStdHxhYvT1Ni+BXF5JbdaKY9QGi+wwf3HlqISDOEu"
            b"zKF8ZJlREQtPMetSDO31fOH5HRgm6OhUHBjkVgqiu534iZajEu5h3uWY4STEFv8mToLBw0J4jli7Gq+tmcGv2Wx08miRuacPafXo"
            b"Fxjx48AIIQA5AtY7EZB2QXmKjOFep390jBBmOCPUYed4QgU7FUBZpTEL2LDw70hAzBeCqZvdR9UietFwqPxc2cyQ+wHoYKfEpQeE"
            b"N8X0FHNwd4W7J1xx2+w7dyXyQoELbW+b6eG9EBM9XA1nH2WU8jFtPjfWGdu2H0b8DMoBnAqGeyMfPLLw8tXfVo4oZ/0p9YcHfBFY"
            b"1u2wbGRvgLfKXrav7meKj3uFPig8DrC4Pff3OGeoHv3qPvy07oMwxMc6zGOvhKUBBys1o1gwYUBjxXmwRlgkJE5MQS6thuDBKeI9"
            b"E0r+FayQ9cpxiCCvWCFdv33kK9CJS3iuxvl2p2oRdtdJ9Zr00eN6N8oHkrTWhIRFeI5LC9uEI1IZgx5L0RYY95KycvSGjZLXq+jm"
            b"sLR0UgT0bKVX8d2EFZIGBLCTWDIJm5FPeI8/hEw+aQTEdx7jt2HjsiMOnMYKtZa47Fl3tCu3a12em6V4ZbXQjCCbZvMHnMo+oiq+"
            b"p8+qvVzSKtRJsmsvHz0UUbAfMzrJx7kkwipkDcXcMUuJMXHN5tx5sCpWi0xp6VisgrEO1rpYyGmrpcEOSKynmWd/IlH6XR0zfEBw"
            b"hqnSnbFjOk26MLwhHM4ilzayyc9J07jYZec9/aFfVhvpoAFBcbeTci1zdmdiOedj70pr6SefmNyMV3w546z11m9OOulwsymqXA/D"
            b"II4EvR0zu6QtMP2sBgXrA0pgufwv8aWfYj8K8+Xsb9TPrqeDVyZnV66UM/PIndhXt7H7hvZ8rvhLkP4vCNISGc69R1j7WAErBMEE"
            b"Bdg6sE4KwWwwzAXMLImvIS6Do9gijDXyymDz2zC/m2af9cvvS08ulrCpFOeWXzdJSSwre7MapAL9kravO5jT1vKtXiuhhYhs9wqE"
            b"Stn0IPjVYSm4vM/YvDhXu89oHJFe5mjrbMSfYiZz1L5xCduszrpWyPQLMLZCnnXIf8BMeO6KcfXK7mvx6lHfA593y3874z/eGZeC"
            b"WO3BxYzIgYLHSHCujDRgjIgZPqZKE7ghgTnOaeBeaScdYiQWpIbK7+mMk0sezKlr8crzbCFl9IJgdtdt+J6P8Aw8iHaXdN6qFJeY"
            b"jBZWfNbL6KNVSX7AFZJtKyOVGm1wCknZEsRG396NOLYDyNtmwrAz+DDZBeKujNFKWJZ3IkQ1VyKqdY6MjW21XTxWDJHB/e08rL0Z"
            b"Tk804Okvp46ee6cUjXeHuSSyV3vMqZMX1B2R4krDiqPzHRqvuU6+Ot9AiT6Wg6cWyVg+hQ5CNReFlqw4naW+NH2bzlJn89K1Fdx0"
            b"slo3e/4xp+QX/fGf91CeLQqWScSDRiQQLnwClUvjhRZCc0fijoBwhkzcLXhFqYirgeQAChwR8Tvp4RsMMdcLxadXiXt+lbA3qPep"
            b"2ZbwurD8eg5jjdpEF3bLSDOfJINNdu8j0aRMfL02zRvt+WTkNGeiL7IwrSRJcOVXOpX/yDFlDUGTRnu3JKCMC8ak0hZhdZpuw3Ns"
            b"plju16HqDN0Ep5ACoMEcdXfZzZLz51Di3rauT1+gJzNLWumI1o9XNp3Xek1zsnot7HFOIkcvuvb1r+vS/3f2nDda7N9v3Jkvbsuw"
            b"e24NOpbQWgtHCDExfdogqNCBMKvAx7yDvXHeKE6NT2LOCBFuKKGICY4M+3egKF/qxMwRK7cAG+M485saMfD9jZgaRvO4EcPebcRM"
            b"USQ34EJjI6b7Ur5qxKB2xX/diPkeTMr/dCfmF5PyPTW3QoZ4IQi1JsTSGuNAbSy7A7EyaQMDkcxrhUnQTjBqhJI8RjjsnXJMmXcw"
            b"KbuY3FXR/ahzf8fmuVYY/pQ04KllVCsZvUAqLuDvmwRqA7Tu0RgrhmfnA90KoVYIluZ8LE/cGD7L3mqs2wJxtjeNX9OtJ7Pf8MR9"
            b"mYkSF1cA2HErdTAaFgrIFwGbFurBy+wmzO/OHq5n4yPwJodtSyRlWC4G3s09pEsvwW+LQz1L7XOAIFtaQl+9Us5Aq7PRIbWfzZHZ"
            b"v/XtlL+4nd4aTjeV++u2+fvIGJIfkzlKJrU7nOfSeEuo9yZ+lbwROhbtBlwgziCLggBDlJWES010TODKB4G0JxT/aOX+Ef/pVwX9"
            b"19Hi8zp+0YIZRKTGVgRadyOmZT+ZQc9brN3WJOjgdjcN/256H1Zcpo3Xc4Bk5hdZOisZK7M7Ce4u17W51iAvVeKBAIVT+X9bllbH"
            b"leisahXXCDh+HWRvoZBuGdl6Y0cKfuqFnTd8T/RSM5TlED3hVdMDKmvs9nFGx2TWwAZ6qZ9dWGXX0tUtDJzT/iHO3Pr6X/9Gi+wX"
            b"jZSfMc+eFczMuJiHNQQU/yepRDJwKQkTnnophAcSqMACI+GtoyHEyplLl9y1ERfwr4O4xcXwUw193ZYvvieEyXlSDZpo43uuOmPf"
            b"WhPSedLLihyeA0cam5zxLn6cHu15/nplSO+dIxeQ7R5nXWTh0I43zLGqG1aW1+MPdf6G27VQ5L/qBq79XTlSbsM/Wg3/aGWGQqtn"
            b"9ePaIguaY+ZjwZVjypES5R+DzX7VXmhE/qvNyZ0B3QeB2YoHizQFSIrNSgfHUdzxK249kQ4UTYLPSGsmJHjCLHGExeqUKkIYD859"
            b"h1rHuiPws0C+e0Vin4LmCs93mwN7OVcB/fpSbtQTGWBme2o6hDpl1RwYbLqqKlfWghlH3r2uc1/iDB8p9GeNDyrv6fOnO0zBRO1d"
            b"2b0je17S5LAUXZxPZC12srrfHFY+na8oQj+e2MlqFierclP+XyvH/1qan+6y/ntPoBL8P7JoM6ybzfLGEvTP2f6Xbf6LwV1Tb76/"
            b"/f+2wZ1SGDsmOVFeapmEuzg4DkCsMtYyS2O1qYhXWsQUrT3lxnLwwlilKAHyDaAIMhfk+8oM73qAd5HooJ/hlMyzbTK32ku2/YiT"
            b"F51SNKc7K/rgQzfdzzyKiikI0bOPtPcy620uojnrt9ituQY+5Cglz+ska+e8FJaSKzpd80R1AHm8NuUz5YAx1bE+w+UgsZEhx4B0"
            b"3F5LfvN+ew1LWIkjXe/RNyoKnLSV87iK79c/afEKf9LI6lG2u7WH/qhEUiwjhRIB4lqJJcPUaEmEQUYSgg1m4Hz8XiLkGWKOQsAK"
            b"BR//WXEQjJvwL9eUt6rFm4Xnvt++dtdYwMi6kw0q0WJxA/lExQPv0OVcyWbkmAyLlWXaVYJrAMIAQUuR8aaIVLOZ0tt1dSteTSrx"
            b"6q4E5fLEJR/aTIet6hQolhDPqWFRA8Xml5vvqsR9Qa75wa58Bf2Syxwrl9AvudR+lsesSB4dy5Uz9V5Oyj9ugvRvlJCeeyqETu58"
            b"TDokpEZJdQIDQiFoBpJrQ5VCRiPQCoPmXBoBEhGl5Stc7aKA/ASm9mhUinWvEl61K9/A4uYv3gmcrf9Yt4HFAITt1DvTMF9uG2fY"
            b"ZQ/beb7KETF11jhSNAtt81RZuNR0q9y3U/niDnmJwqiZ/vMKysrHVQoVaIEAPj2ID2x4cWAWBEUdWnaL7ibpKa6scULK9ogWkhA/"
            b"tRzydiPzsbzPCH6CvQdZTW9wBXuFSt4Hunw7TuLhyJfzR18X7fkZUOwXeph7TxJVaRCqruaqt1lPbo4epmaIBw4h5jITq00AG8uI"
            b"uG82gQUijbPeaWwtppLppDjhPIO4cZbEUsYC+h9THO5bjLxTakd5FInzuo12GbDuGuf7an6mkx7EU++ra+FzuNhX42pi3An2xstL"
            b"UXFRyxXXKPA7CLKhcja0u2pUh/Ri/PPye4wCue+t1aEjsZ+0aVTkgK9IC//7G+hfneF+E62ZTnrBFhC3yALXNn6LNAsCc8kQYSHE"
            b"3KWVQAZrxj2y1gqqmI2b71jo/T1oTXyZ+t6S5hnQmqNyzB1S/wUKs+OmFsgkrYwtFqEpJmMsRcVirSCWU/mZJxJCM63IExbayNSk"
            b"HCjKYOZgx04FEEQR/Yp/9LIMu+vwjl/8VJ8oh5UNPKIE90ek32ODyEpXUCKBbbrD+9VM1X4SoS0Hfs2ILl3g/V33CrV5tbdekWfl"
            b"BRG22WP3g/HzH4cJ+S5l9hfr4bSiB+Iko9/UvfkmAKeWikqngzJWckEwDTRYJJC0HqhBXEnmMHagfKxGPdKC8Lgnj3nbY+WN/SKA"
            b"8wIXtErLy732ZkJxAQ1aizu2qo0lh2J+wgHTKtNKGg4mGNuKQXqpxLkDRh+G8jcuSSoCxp1J8GCXnGN2duqR2l5M+SdAo3yK0u3c"
            b"e5dqr/Uuu62lG9l9cKVVSBJ+B4vzt1zaAe10Or6e0EIFhL/9dtoIAgVQtG2gS2/0SfJ7JgNwoiDrx0UgAFc5DleY9RUa8koioOL5"
            b"zx+SXAl8cXLzei99JEtMj27NtpcWp0HJXVSkPHke++9quo9Gn8M+7tlLUwIqpqlk6648jws/xwh5CArbuHUOgXKJmPeeE5+si3Uw"
            b"Mu7sGMZMMvKjSi0fyWHsKo3JD8LP7w14Rl2XS/j5tV0GW3j5jvBzeYxCnsPP5wz8Wht3kDzf51G0XRD6u89exSkqbn1JI22Adovm"
            b"RgchRSWNXrXLDIj6SrbPorvJHFwuHTguB9YfXWcXXWKgYe4/y6WJD8p/ked/TI4dkeeV7etPZd8J8lzrtOPnwopg4n+9cLFCDJzq"
            b"ZLSppPJOWicD59YRAEUVWMudJt4EME7/L1M170Mt75h8vqR93uVqfieR8UUHAl6rqyydh1YCK7tFxFxmpTArKaO9KMuEUZmjsnbK"
            b"YkMPfQPl2NBTtVPx503fHEHou7P2R8YQq/07LFmXsKngVpaZ1eRodbbODqL7tzN/n//2apr0y9R8Z6PvkQwMMfCcI2qxiTt/pDSy"
            b"BLNA4tbeGO1jEsLGkIATYh4LLSzhgWAK8Adt9PGrofpynN4bDUyVN8gVV3FsS94h9HRq3K1yRzN4STFCkl5ahF1WzXic398T7Dok"
            b"+jDBlXJ5wYb2GFJJSlzjzDAJzK9nGk+zv0eTwG6X/yP2DGWDL6vNvqw2+7JKYbLf7B8ew/Wz1z5pf8KG/+vWDDPdvvvGDJ/c8weP"
            b"RAhBOCMohZi5LGIxQzkqsbExt6HAiSI6FpmUIh4UKOKMY5ZZiTT65f/cg2E+UAe5hdd8AsO8je2cFYeryjC/MaGVrN8G1FyZhqVT"
            b"Fvm9O7DKezjZJ9jLlUjf9HJnIn3PiseH2qkreBHbIJy4ghThGxIgKyXWylKhNreZKDZVkX+cdvY/AdU0IJ1hijpLRdzLs+A5aBwU"
            b"DRAs0Y4hg7RxXoIlynFngwUVDDBNgQf/DSUk/Tlyz01pvlvl4Je8FsXDGpZtxedgZdh7MT4ymEznS3XtWKuS4SpfqAt2stR5MQFy"
            b"eDse6W4wAc4WlzmZMkpRdcWwfocSmjuijCN0VWineytB78p2PJrCw8XcHHarsGrrzZsd9qykhWVJW9MlM8O992isOJYbPf0vUux4"
            b"JoT3H8nd7TmVU8MsgZhblTGcobgBZ5QAY0ZobgWNaRUoNhgkIDCCgPaMWxr3PZ4L/+9In34wcX5eRfW2tNEtN4WhvOXrCvdZVr7M"
            b"nxMf9TSt312IywelqgKazo7ZJJOStcFe8h/O68eca2JInIOzUgpUrIL6iL4Az4Elv3NMyTCrm19fiS33k4ZX0H/hO/8Zkpu3b6qe"
            b"pm/ac8OBUXOE/R9UOiP140w/a3hLd15p9e+uhU/n7Ym7PmS/0qffI31qZKyfdTAWC4dkGqFJaZDn1iKTrGeIp9JijxjTWnEeq5T4"
            b"jSM21uJEecX+OGQD/qT77k00wm3B/3lHdXjXBQ5hlAQ5MjaXZzCuDMrFZQsFlt4J5ZS5esVsl8gWKwv00TDzBiasdWcolzLHO+SY"
            b"Ah+7jXfI4AixX7eq57SzA9L3lGziq7MbbcOT0ilulE7fainn780TxMM8C+9ZvNGBanhaowA2bNM2eTSaz1q+Ksmrh62FOu1pDFvk"
            b"39hcvmEK8yNt51ly1mCJRZpQywMxRHMGKmBBpSDOe8PSyMwaYVwMdIJJ8MkaUlgbLNf8jwI7LJEO1xyHFwSHbtJWKtx20n+FWui4"
            b"D0PBzPb6VI5q0LsSH698J8XZdHvGqOil+E6g2PSUkx3K1Ll41OKrIQ1d8FSLD47G49CnGsX4Nq5Ge2e7HJ/6khvvY8DBWPzCUnAP"
            b"loJ7sATZrtoYFRLspeDen0ImaHrAFRXl2ykDA3rAaCG90tgIoaVUwTtkBNaMIUDExmwIUrBYhhqjhAgUY8ydlN5axbVQ8D+v83xT"
            b"vvm2GjTfE4gcfVI6QRO5eVkNzuG0bRycIiJ45qPSS4jg3WtKoiqcHbCvdjaFtshp03VinHJPI/qud8rWxY0/2WvvlBIGOR5Y/Qst"
            b"VapoY6HUwO+2iB9BtnBFwMKNxxWumLG4Y8zOHVWuitAVxZ9OSs4/VHDqHxBzNoYJhpS12npLODgnXEzAjKei0yWEFrXxT5bRuOkX"
            b"xhAQJl5cUAY0jUF/nQ35usq71y2GXgHu867mR5n6QsFq3i3u50u5k1r8rlBrriJrf9V3OuBHWxcYvwGaHZaT2tFroG3QbVk5qGrH"
            b"51GrxAzXlK8lVabJu+e8qAtMRYksXeoDhNFCc4eLSxdVohmmQu6f6t6fnn+s+fMv8W86keNn2AlohFrgiy5XUJF327PBRRu4fXXq"
            b"f1U9+vUh/3EfcmODAWwCo95LopCEeCWaxsTmKBeCGLBIJ2uVwITTsSTTYHQMC5pJZ90vp+Ij3lYfpFT82l/9ffZXDwv0uY7ClbLW"
            b"lebWiocxY1LM/q0v2n85FR/lVFgcDKVeQsy73FkknECMgffaaQHegrVGaisDVx5BcDSWfFRzzZHHWpB/bEgn3jENv9E7GVEVs5K7"
            b"ajGXgru50aPpude56uxGyy+5hd92iblosWTSyohsLk2WE31ce1ndE3WdizlMgMUrFYn8SR2w4pO+JjfUMh42Ux2B7aH24XO68cYx"
            b"PvReed0hXlKR133layryfAh36NNM8HJbyJ/D83jQIfmQb/i3dUgsVyw4MA4xqw3jUliJYjYhlhgmhTdG2JhuRUzBAqxCIFGsnqUz"
            b"jhsC6N8Tr7mXBu+K1+Bdu0bi+u9+oXQjSiBGSp3vvx0xKt7cyPtyAy7zPUDNOHM0R/SyYAOUF2XsBqVNFKs+2l1HNl59CsOKkmPN"
            b"YpsqbHsHKQLyBFAcDoCTqjW/nBfDN+Vs0kf1zIllLvtKKzlt2slsw+HLUj/e4QhQwRGargXv09wpwlCIvb8yNncobdYoFQIwZrmk"
            b"zCkTPAvGIi6UQEh6IpF23lpNBDdGG+WC8lRDYrQ5/LNgry/KV6+6sl1ZOGwjz/rzxjbybPJNxQfu437zGUsMA6JwP+l6L7GN7YJq"
            b"S9Wn4FuIYlLVlhifMjUlWcNI7ttuJ/1sq8vVbQ2KDe3qMlx40T3IVwOE8EGq7NASh3sZv53RletOgQxDLR05XWvKBZTIo737re6C"
            b"UByrKjIHb9gc6UFVqrbP8kp1ZNrmycuh3UTUdqaQ02rlwGOzrD/EkVBsX7vLju8DPNl3FbG3O77WJHkczhmVoAOxQJOWmSdUBCmR"
            b"ErGcBe6ko8YZjZELBEM8hDNCBDbwywZ5nw1yD3LA94WDYElwRyU8v05HoS03bDHBjfdWNbtqj7grYvZsYIer1QFVQg9TEZp8zhIJ"
            b"jCI5wEYm51cb7TgN9wilU0DIdEiZgtNPOS5C84Py0DRfFnAQ4qSfnP6TCzZ1Ds/68CArfUrWNI8nR8XgcoXvkkTIo9ngqunLtpEN"
            b"rZrDtOJV06qFTLvZ4LhczNjSV9NB6GglOwaEvWws/xJEvmmpiNsDjx1yqboxlBITeNwhMCJl3DoIRZmWEnseJEUcgbXCJhszz5EX"
            b"PLB/Z6l4Us7fyvHv6WbO1hR+4jAw8AFVx+54ec91JO4i3xbU79GCd83nxtW+QLzNTHzBAR9wH/sKx47ue0ehwfPvAtqOYJjS3eJo"
            b"ZMf05MR8B+UQTgXDrxGK6YMqoe8uC1xAXGn5U8vyuWOuLJsG3uwa+ImPlttTfjyfPavOVT2rdNqzL8rEzbz591V856n2F6AF/71V"
            b"I60XPK4SnjOruHcWScrBU4E8IsYgT43HGJj0RFnrVJJRxpgpLoj0wL/BsPeB49pXndPuDSTPjsu63/L+dHBletYvOmI3HJc3pJZG"
            b"Yt5MaknM5pw1Y2Y3Iz/80W9rLh0Kz1M1zk5E6Zhhjuvfwn/zanOTrnybfmKGeXvQcRsz084S/+6kMv2yntlazrrycqmWJJfAkKtX"
            b"KnuNzs2Sdo9Gs8s/377tG2aT3yaW5EATbTxS3rLAiNU61uI+aKli/W0DCc4FGryEmGot8oSagBSRhltuBBbPfS3FOxJ1fDOqWvFn"
            b"YM8WU6TItf2beOUAt9DjxJsFHFm5wG0mb2KwgSu5Vu4xZa45fzOSaTE57Z2h+BgaXTMFz8nCBVcQVX8MeDeZ63l9CVzFtuL5vA6Y"
            b"hrNSTB2GcBsYrxzS4QJZiYtL3mkhtPNXZvF3iYv5hLuH3NvucBgpkbZFT/KnbNWQKkuhvd+NG383XENCoEaBNE9I1VNpnsm6iy5P"
            b"/MbcxOgv8Im7ZBPecIr7oP2lw9YZYYUM0gUQOEgjpXVxR6RjRQpxL+8w1gZiJkRCMKRVcpaLadEbw5z865gtz4TbP8EtecKl+RFZ"
            b"pYXH+VOaypOOA911ijAhRxebX2l+btpGKS+fKkXs1Bxii+MG2SV8KbtUbrfQWnAF1tsbDXNhoxKb34JhimXLhrnSUCrhbxJbHoqC"
            b"XtNXaENKPFvUM4IiW6KpXxFWWvUjOmlY/xJb/lNii2PUEZMuIe7bCHilwAWtARmgQjOnmAEvPbWAmObUWW2MpTGxCK0kD38T3mWS"
            b"XKdb+5u4GLql1/RTjg5LhDzHjpRTZX6E4jNDJNJld1UWM+AKdUno8DQ8glNIWv1AAKcDQq+PXqT3aeMCzuyOWAPmnofnsLImENVc"
            b"iagKr/6zycHl4rFiiFTwov6w9mamg+Kq6XFzUHx63sv+CHWo93N0vsNxYfmq2u9DvoES/S4M5hkW8coUZG4reqXrcSWPenLSr9jp"
            b"Kx57RZgpNf8v9OXHoS9OUbAJqGWFIsmPRErKFLWcyUAlYiRhIYXA2HMN1EmwPn5VYzJC2kiv7u0JLjYEjyVEyCuJpAt9pIcqRLfE"
            b"QyYl//Be4yaDDGJLrYaSPFWU9kT02rx54tycBlKk8mLeTonGZu7mSEK3G61lnNSkIz4C3tWgLxUD0ptLRk5o5HlKaE+ZovLbf0Ua"
            b"ieBPaum3WfP0JKk98XCFBpljthugx/zh1tj445WRljrOt3SRPojcdo4oK7S0lCgwVlBGJEI8GADm4v8kkgxpnLjbkhmEPRU4EQMV"
            b"8ZYJ8R2djI/krkfybm9rG01bjLeS1G1FuNtKSRN6Ht6vEE0k7YoERkcknJx3RuSrVS/6Ey+IfCdscCDy5WKcIERbO5EOAp1ez5V4"
            b"T/kb1JquKH8w2z5MPOue5sv0HXqSMVeKcdevrGyZV2zAxnKkejgKJM8Nnv7KPPoJs/pv4vt55MBgyuKujJrAhebKaI2xRpR5YTg4"
            b"a5UFH5LWGGNScUxFQNxi4ojl/xsGJDdcQx71UTefjZzBCG+QUQvjpBRWCC7x0Sg795wv80GXkye+JeXy88p2EwldwtJSwqjoUWST"
            b"32kJy0vZ97uRPJLrlEvosFzqGsulw3OBruGqFYBnkIUKUVaByKp/PbjV8teL5PNeJJ7HwpWDJ9pCoFQzhuMGW3nglnkmJXfOc4qC"
            b"NIqwEL/qgmETjMSeUCLV3zeWuztz++yAbNbDraZefbt10pytAK23mrOl9fOgOYsnp360Ziybs8slZmzOsuth3bQ3e0gUtUL4r1qz"
            b"+BmHB3f8lBut2etLmrZmv9W+b224t2rNXhXQ69bsSBVZ+468mNa9bM3+zuu+qTfrlXKMCyHi1zRg0PE/ySWaIco95kwYFRD1kqaB"
            b"XazBsaIccUqsoPEBgXeE6Er5+g3yRysFjg8KH+WvZmue8dCEYytmYXDVaJaRlIy7gR1u737PlDvfnMwHfND3QdJfB55S6evTql1H"
            b"ZMK778jg8Y1zVEq81+JE6a5zFEWnUcjYTIqvZY/pwiwvrZQ2gpMM56rNpR94gDxS3JireX7LKy8e/fFzq1JZdPC0m1OoD+rZe2V8"
            b"kN5JAGSsp1wSp2zg3mCntCXayIANKMIVYEstQxhhZYy1EP863D+ouHETLfAeMXsKebhUTpNvGfBNeNE1e2AUMn6kmrHBw7KkBVJH"
            b"cq4Njyb90RxaSNUchBqz31Iw+QHW4xncoJyxhNZqzOiSnLFSYz7T+JR0rbbwdyEHz4gWK8yYXCLQrukUqx7ISJ84/vX8x+Wjvb6V"
            b"H6BP/8IN3ippY5lqHaYSc8ydDkQYrx33AWHmY70SnBCcAHdGOi3BYU204EHoWOUirN8hwh2blAVFY9EW4V9oiFx3Q+bkC7LRL2hL"
            b"J8jA8y1FbmWmnM3aT5rGAR3gLUsjJR2c/x7QnvPnJLutM5qACIe2w8rXuV+X5IorVlE05JC4WWvll+Iy5QLhMx6vTU7zZAThg3ZB"
            b"Ss8D3dDxnyKpp2SLR5nzGQpBDjisWnmixWHtLQFaTdtoNW3DVWO4fjybk81bAn/ORr8ktheEtGrb83rb/hna2ZbLAjLgrbZxPZck"
            b"bsdpcEE5YF7HFdt46wwiDMAYao0NWsv4P2ON1gFczHTfkcvWXd6fbvGO2agd8tBSIFNAqErFG1Wm1T/LW2jMaurt5ShrnoP4Lr7J"
            b"aSuNP0rJIV6kZ5Hix064biG2e+z8PVZpd4UPefeLMR0q90wBthsXZ4Lj7c3nM+ZAgYt0/KrP2mn45W2HxIzPuqwtPLdov8lyp7u9"
            b"514Wd5VoPmMKbeU+lleSrzrHvikCnxkU8imkq5bRlDecnmlRjTtbr7wGL3DayT0czwpuq/RLi+IC7RELlRBD3V3do//SVPtFXfhv"
            b"4/4GEJprcNRryyAg4QkBCSEE7zQOlBLngo1fOayEIZJbDRQxryySFrD/LJb1uypGvv91IznJY/iN+VmnEcPX1R4cAxZcdrti//uf"
            b"1nH5GnMoAN01N0Ujw9NecI5KfzDA9mGXmCXGPSL9lGezduusdsF3kL7bYoSUaI2gu8tL75YgAmxva9PDTZkNsTEma71jLGnv6Uw6"
            b"zXuxhaU8CTvy62xnne2DvIN+L5nGHVf8fJ80WPkmp8lPuczqaf4qFmLv9nL97MJFY+3PvM2aCpyqezajGmzp9VYj9q+cUv1Hs6g9"
            b"nxIpgAB4HGzMl/E3wYjFyehOJkd0rlRgyniB4l6d8MDiWq2N44wTyiX6Gl+YXYskHITtuUgCvRRJeKGQcANl24M3b6Nsx1yLF/DZ"
            b"2vMe19Jbrzzva/gsngFdp1b2vZ/PWv7mzI7Dp7D7wGNG8CQtT859i6Jd9hKn1iSb6VAU9Z38zqPjKMy2ICVscxEVpE67C+PTHAao"
            b"w9w+VGHIX5P/FZ/7n5ZheA243bCQNwC3L8UZvg1wGzh3EjuKLQ2MC6MFdcgw6mLuFQIzz2wamlGwMTezxODlFAMPHlvugvqv6A3w"
            b"qtB91UV4jRJ7CRG7SdE6VRm+bm+6cLj7EANiQDncZ0BMSWJk4IkV1kMhS9BysaoxyeuTthB76M+wGr7NDvS5tecL089TDKchk70y"
            b"AZ2bhf7xtew39Aa+MaWGBLQVVhPrqLIEpJUUIay5DFJjB9ZwScETp5yJWdTHr5tkRnsrPeX/cbN2yKLXPLFBduseVWzdkm3C0usy"
            b"/V+PdCUvm7J8TGYzYcWBW7EtBftCgKcE2rn84lykJsUkQ2UoE6NTbkZWkof9G6TgopKoRiNP2KuVXmwnfU7bgkH2T3b/yAadrxxS"
            b"hBv5aUPK99/7pJjfcGM1a61y8mzEFxVscV/ijD3vI2yqtucmv3qSH8uKaCvr7kKlEVZ3b9eCvLBJ7vYPR5eNrtA9OrV/Dnms6RTU"
            b"kmLfQx/7vraswNLjWKxKCCSxx4SRHDQ2SMfCBxSy3BAB1FsXCKGOJvUZB1JSaxX7KUlG+ue4KKfZU3JHlvK0iYPy5egmWrkSPDql"
            b"u59wl2HTP7FzMjWLileQkjXfAKW8HgC1ofnvJF4Uogvn4Fvyhl1qEmhX8tpNiEZ7YdQU9DmQiNJZrarVSTAtcfGH7Ji6Hf05BRCh"
            b"jrFc5dDcLB2sRL0HDnjqKjcHT8GS8vXaDHmm/jKzM66Mjf4eZMCWAmdA19djqQ9KLQYtNIvVZVA4SIJ8sBSCjTstTBQKAUkjtDYc"
            b"W4xC3OpDshICIC5u7olW4Tt28F+sIW/sja/UAW6khmmJ+AH1w77GuiflOOk/TMvOV2KK3buXsOxQcQJAa2GCscpbAUBrS6P+U8uB"
            b"m5PDTdHCEghZ5pDSCmULlWLLRLmxhDOCgTRY3ukBJYzk+MIzaMBaUOkbHwdQsf3uG4rW07K1YAkeQFqz/Ruv+qvVk81YjtPaZo7T"
            b"inR74glqXYS1xVzh254a4bTn4NZ03Fohd7B9qBXH/3glhAa8erQKjiKX1EUuqTM8rYtc+iMaCbdBrsGF4AgT2GjKiDdKY+tNoALr"
            b"4CwYGYw34KQVLnAdhBUsrhc27S8pRvrf8Qj6UvL+gpXQI9ebiS1o5XvQk7R6e9RqMR2cOI/pHCF47AHjEXiWTrvbnmJOB+cisa+k"
            b"nc3mdq9AqGzUErb1YXFYCi7vc9N85wuaEYND0mbiE3+KZlI39R0tYelt4xKAWTOpIzPqXQmLJ3vXJSh9fT6h2bBS2ZXL7vPVKyfP"
            b"4frRzHiiokt8wHji1wborYXBC+1wcIoLMB6CcVhSJVOSYMhIbsFx7oNjUnMsEq7CKoEMF9TFS/4OIZ16GjTpeixpu1s1fMHcXVLl"
            b"1BreC3OA7wS7uynNTrG4WRa8HY9NlGlH4l19T/M2cWN2fE2p6+80nyJFEMwqHdztF6Ae27DOpnuj7E5+M5GZDAVAvP122kWa5Nr1"
            b"oOyWJfcJnOEpQnfWuGBLrZoirUhq1UWCb6gjzNUOqoeHucT38sCOShvTg0u6ZUJxOJ9cZsKZAc+Z4+acLvRx0cUQKCaCWAVeWS+t"
            b"BCs8KKkJxJ2mDUojDDwmL2s9E8EzSGAxSzQ3TCD8B/Y07vtJtmUfbsuqh2jTfJp0YQeBv+6iVg4hz3sb+Yx7BUkUXvRS8f3yF980"
            b"NMaTxshodt8jwXrAci6ZswXyTMJsH7G1H/gAD5tjt9CWMLdfyx48RYalqHLZyb3+0OjZjpld+RaYflZ8vvUBJbDc5ZfaGvkrfL+h"
            b"XAgbp2RM/WxrEPO6Q8zPhjOvuhXtkwu82dhEPuZuS0mFsxk9Tu1+Gxr/QUNDIaQ9MIaAaEqNUQKQk45I5igL0mCKNUZSI+W8DYIG"
            b"b5W03gB2SAfl/gpixUOdxyHdvEWouMdN6HbUMOEJz1QYZ4rgvQoja+hnve7h1r1AuyIOK1OxjfY1dHE6zDIcZpAdL+QuMUXubBPY"
            b"uxOAq75y91mSQjRLM2/ALXlkuNR8SznubaAZ/f4GwbV35ahOvvKhrFrE1cNK1HF7+Eue+Bx5QiGMvaKcxLoYCChtLbXcegfS81g1"
            b"M0m0sJJiJWO9pmMeDcAp0WAsIE/Rd5AnGhNJ+mEfyUX+HMgED4rZe8D/Ozl0paI4nVFN7OMPQO7UInKVosnNLE2eJmpyltEgBwP3"
            b"9HuiilA6VP8wBajl0JzQ5nsGGGpuciiblXGkqLcAPcM4C1KiLIHe1/TjmUtMuY6xK31e1bUl/Du0jOzZhp+JoXdTPziZwRsk7Rwe"
            b"8lqIofK7bJ6wbXS4FeTNk5Uoemli8JN6zK992T6o6vgTzI2mv7uqp7cN1o16+ls5Hffrac401t5waRgNLBhCBAtICG41xTYEh7Fi"
            b"mkqFVODKMkiubI4Sy+LG0bwj7Mj/7wIV9xVCyBUw7kNUkMZtJqXaUfzmwGocTBHaGtEkyAEMErcjgy1Nw2Rjr07b631IKeElCp82"
            b"EZXo+QK9kYLzigYn2IN04e0KJPKyHG+q7gdPjnoDm6cKQDu1kCm9onowKCHv4eI4fWRLfDkhO8rpUVNsZG5cvXIqh61na39+0fw+"
            b"JO6T2o8xlcVal2uvsQSjJTGUYSmE0TRhgQlLzQUuwAUTFFMmgOBcp+dcYE/lcxgwWyrZXmpBrjVsxXoeBq+UbJ8LSFYDqjlXKk+m"
            b"2AnUmvKpCpJ4N0U4VuoW0spyBNmtd1Wjj9iBaesEWRaN7u4WI7GVxOQkQZ579v3M8nWCrA+aJ8hSsgqK6rs8o7tZYoor6bECDm/V"
            b"SIsHFiXkTcnbZ0iDK9TAql2wzolXCoyv8AV/kOit2L7RIxr4fdnbT6KBY+qzwQruEBBnqY0JUGMOLNaBklmuiBWMGR2UECEZKzId"
            b"uA/WIIESTw39qOztnek//qRu9wjWwtMicM5h6416J/TcKZCs61bCDort0tiNSfzYpSSb8cxpRH6FYJsCmE5O8OFUXovQjkS6ezq9"
            b"qODEBNphD6f+7/RDL9K76RKAIA4zk5/p5eTg9DZANuWy81PBh9F6c0SKI41W2FPBWsKlQOqRbC3B9Ra7fbZDbuvn3bNzPtY9OSx5"
            b"anue/PjAHKyNdxp47wzr+6fL1s7gClU6nvGCv1Oc9norbjkmOEjMtEcpe3ClLcQSNKZngySxSTXMB+MoMwZrSREWgRkR/yacouaf"
            b"kFr4qpMkf0O54fYG+g/QUbh2kuxP/DknSfQfOkn+ekj+ii98MMlaK7wVIr4rE54wKyRj2kihsA0MxZ8oZVmKPU7YMeKwVJogQikm"
            b"3vy9yLFXnmP0kecYve05Bm3w0nprm2X1p37lUEYfGHWVbyAdjLrKesMQkv0hefVo2hMpKC9SV8ZevSFFDo0/aSWMxuuRX3fIWXHv"
            b"S8JLPYoX1MByqt0qjTKyoyi246qWRXMpJbZcOk+wmz6Dj6LtJSxe05fAZM/IyTV9jd7Eg8nGG4I2Njvz/kWtQbYT41Z6DtXDA90w"
            b"kpt/MWT/CYbMS0EZVzL9X4gFtzEceYMYCgaRIFBw1glGNY+bTmcU5ppIQM4jLcN/p2n2J1u2f5PJ+i0n+Pv18xeJ4137+btd3bMc"
            b"5J/p1Z4riSfCkfNqGkofoyqneZW6V/V0mcGeepH8pnjkmI4/LB/569c+pFrABoM1TgYUC2kCzISAnYhJFXmR6fpK6piKIaHTEMMY"
            b"giIAJmgKwbxlTMEuutAL6tlPGbj3Hjudg/m9nMwOpTKxF3GHffk1DhbWONgXNOXnFhebzQ7edYPYhTY7ynGA0KkBfPpItvMtUcIo"
            b"kltlufetW2FMmd1nkBSbJs8YkI7bx2RPCbnJBzWh256I58wdH2BprAOXfg9QZB/4Cdetnx0SYpteefusIj98oIn7Ufbtc5edV1za"
            b"T7rsKGAIOxWEsSBsMFIg5ojT8T2Vkkhjl6pIgRmTFnnHjbZCKWo4j0FG6F8c7QzpMzWWnWqTk5V+z3q7PGh838Px1sMpfgaKPduR"
            b"94WJHmr0jHV1LdkwfGwpKnVYapYCRheCE/kjK4UmYuJUvuTbmjLWpuXC8zswTE7Pi6b4xQeZdGgdx3cpx3FGODvL1bPWvviAyzHp"
            b"9/IlgO1DW7S1H/qq73Dth75N6KAe18GFV/rcI239kFaPflG1/wWqFpTCXiodArdAXFwpPHFgaZCSGC5I8AjRWBbzhEPzhpBYEwvh"
            b"lNcuECT/x+zSN41JJk/5lSHFp1c3uYLd6+bKpgLmCAqVIRJE7bqKqtHNubLHYLOBYDpTCojr/1IKrZeLTOsDK5RlVWvM9N2FFMMw"
            b"nXbfXyypU5QJw7uwJVO9cNt4nemOcuC7ULJ0VffBD2v0l2x8evvHT4lse/9VHrzf2on3Vcb8NVCvQbQ4VrwglSfMCUKlwVpYhRXC"
            b"KpntJT0x5hSLiZBxLqkNjDLgjhhLUcyL9o9iDvywqu6MJIBf2UnMd9r3TSC2JNsUeX3OSblpB7LWtgtNHMuU+zo3VRStrh4uxbs4"
            b"lM35KWveRoosxXjgflsNdDyRvhVbNiNCDQLodGaf9KLzzKbwCbxLmks0F7Dofx/pYynhExjF5Kry9ZfQd/m/DyZlV+7o6xz8Apt7"
            b"5GDcu6N3OXdOCT5ZDK+c0P946u+XAA/vp2aSH5NFmgbjLWgjGQcRgBsfkMXIcu9tokBoY1lgRkkBKpaingoDwSKFjcce7L9h4HOD"
            b"l3ur9zqVUxgL2ifOPJ1QgTp6NW8Iy6zc0UYsyE0/nktXt+1aBwTf2Ds5tvbTTytfb7oYIimrcQn1CjT8JlJsehOK4DgI0Kury9G7"
            b"8dyb3pWSkljL8Ef6vS/Qu7yB7/J7+N1t8EVwPQYjuEIMc9o/LOfmZ76tHm75+IYU2S/u7PEgDHOjLOFOyqCJUi44iZgkKiRWLdPe"
            b"06C5l1JpjZ02AWNARvs0EPOG/+O8i3uEipvsjE/zLqY5EFrY11g5Tp0tqz38wpJ9mcFlLxl5h7pxS7xxbpRZNSaW7vGNitmVaO6W"
            b"4ZNdEYP1mzS/7BS5eWi+ycJILeJnWXpubymX9pZXTYiVveVKSb1vQ3RCuS/xCb98i8cpWVqwnAdLFaUuDZeRCDIwbZnmwAhgR1P/"
            b"wisP8V+QZDEvc2kN8YYZ/IM+bD8++dvTy/4nPRuR3TEj7km/a71xUXoirYPQobIITTKhx9afTzRc4AJwMVPCyachh9m5qAvPySlv"
            b"Q+L41k85bIT2+cigf55C0u+jdTKC3Ze+owOnqH1BUkuNnOYTSzdXgit74V2z/luchZ9SiQsg9oRE4Ar1hStgLr4tWj46FV1QiflM"
            b"l/xjTYmPjtHqQndlxfZBY+HvsmJT2EhsrfSB+oCt8ZghzwQjBrv4HwNSUCM0x5xaIBgUU8ISgalK0uSafoMAOf1GIJi8aTnUyw72"
            b"aNQFFLVWQkBFc3mAWd0pBm+aZNyj+KaA+HaiWhD3P5kuUMAp8NjO+LbMM+LaOlRELRQ8gNGy1Hk8SG5+fHTPkztMYcAoqBwdv3f4"
            b"aJcDqroKEzP7HJsvKCZOYOcvY28vd+PKFFN3IZpF5HtYEivq2pVRGzQ+77hxvsQV9qFFrh2420p0d/7wZFS8SrQ/hEXbq9oKrlAK"
            b"3BkhojR/14SIPQHHX2uZjUyQat8q/RhTLBfOYW1BIsKMdyLgwKTV1gsrjLFSxOogMMuQAko5gOSSW4GccFR/R/f3mWju+5M12czW"
            b"Ohz8B7zcum7CJYtYfqAfvUDa3ueK3e768r3AzneDNgHdSkOiu4TCLcvO7Yg1XhdTFkY2gdhAcArT9mqOD26Qhkyh6YIYQi0lAk4Z"
            b"h24WlyPzXcTdXEE9HAfNmxr5Dkrwl+hu91M4wR9SG3tHU2xdQsNCaeemu88tbZ6/Usp3LKkbc5+fF/mddTa8A8EDsYGGWF374CQh"
            b"yHhqGcbE+rgAxPyCSFwRSNx8SuUxMVwKoSTTmvzPiTrm+pDKxry4WVSSIO3WJ17INUIKuCHr2APAKhuKFwCwFchit11XeEp47rEJ"
            b"6n9LzFE+tjge8yEsdctgmV2vXjn9JU6L4z1/wq+W4zMYGgEuGKcKM28gpjkdBBgd61oUuAcaMyEmjrn410S1TkpmMQdKwq3ByhkH"
            b"f1kf4U2niNdb/4WbpbhhqEheCouPbmCjqrg4WWxsVNUe/B9EI177AqdxQWe7T5H7glUEPLWKmNLfZlYRD6lrz/oFV146c4TtlXbC"
            b"yiriqjMwPryfJ/9G58j/yB9yT6VYUKGElECC5ogi6WgIyBNCpNPUUcGNJQAOO8MxxCJSpGIyptLAFGV/hbfOB7fekx3/5A0vN9Pv"
            b"pfNP+vDgoklA5IAum/GXSYYHjdiuq/Sab/yeb0663BxJVHHh3C6btrbvfQpNwQUKptjg497JWbB9KXkP/SWJlBw9827PYC5Oa4kD"
            b"XlEVeE3jrZ/l2GLozneIV/1UFiO0gzzWPmvABXPk7RR9sDu3/zrwfDKzEgUCQIW0IdciIBOIckIgymKGdYaoWKsazDAX3HGitEj/"
            b"5ghGDpj5qSL1i7jcUr4ua9cOYLC77fLz7zyTujrE1+a2Wwq0Y/4yGkfmEIwUbzbxvaljlt1QKR1hXCdGNdnP55jebndoDbyurPMp"
            b"suKZVHDsu9Hu5kBfD/IaoYZ0knQyVQZlh8kvfeEKIUYORTpFimh0EJ6OoJ5qwdDKaJd2Brxz0124MN29GmrNC8jq4UdMd39i6HRV"
            b"Kn7naGmSx0ISOKDIKunBciFtsAQFF3clieSlqbVSYG8VxyEIAZZxq4Hi4H3w6u8bJ92w0L05cYKtspN458xX05EhT92b/dyfTkGp"
            b"LeMP0bmb4x+YEJGie5XejdO5ZuKoAHF4nzFMG+8zvisxjPMbuQXHn6VzqloSwfSgdL/lgJtTovwp5t/jzYlXCSNZMoyooUe8nEXF"
            b"2GMW9c5sKX2HngkdyMoNTDZmYLVPGO70cFdiBqVaPVUQeCODMNe/rbQMNt0a3skhbPyIQRphlyf/HTj9+MCJ8KA9QlxiA8RrJLGK"
            b"/0s6SYIRKZTFFktEieUOGeuCdlY44BxrLyxz31Df1jmrWyrODL1mL6SEdJe9MPjL0Hschw2ERHv7gQ6tdPh+HZtfdV0yNtf/ZexX"
            b"h8v/eexXfwEj9ktW8K/h01yCuUiVrPvf1RLMdXy66hLMta1mD9Cyz2TLVxLkKxwXW+K42BLH1WiVH12D1cNet+Z7iQpHUsX0MH3a"
            b"SmpxSA9d92WLU8+W63LxN8nBO9jrzLtzogL63u6B1MRTycAFpQhLRuZSxLo7luKCxYybZBUoE5yldBsocekAqQ3WwbG15/k3EBXo"
            b"Txfo92i7R41+XaG35Kul5PYdAcR8kpyq5CFPsKoOFyDf8WZv04rzuyY5ha17MdNTeOHwcwiS8UExTOFdxxbXYu9zDFaJyhiFSvZ2"
            b"rp0wkb2lyyI839shfPtOL/ehCthqfsX+b29C1A2JM7fOlcPk3gg5269HDseDAliVaKv6+H7O/enC95GQ4xcpu9/GUCDSeCsR8lYx"
            b"4hkO2HKBU2kQr0R7RTUOAmFlEGcoaEURAhuoMcJyQ/g/pl17AgTylKjHBqDck8PbjpC1CHv6WD/2VlrnpV+x+0jAerC2+0LsrhCw"
            b"7gbQHKT4/rGTdWhlBlzq4Ppvp7X83SrWl2tBjqG5sbL1ik/hyVkOjFHpUt9UsiUPE+Aqza1EbN8tVaFzwJn/669s7T3ZWuLBMMSw"
            b"90xrQSgoG38yzQARoyUiMmiJGdOKgTGBEcEMtiYgi0IA9NfJ1r434p/q0H6DAu5i//y6qzz1dWlrxyNVTD+XjNDfhAx2nu2eVkcB"
            b"slmLV1UKLn0hnuZIG6h/+6RUlTzZ7Jhi6ZiuBfaiu9WenV9ZCU7dVAK9Xlh9GGk/KrojEXbX3dM28uK+cni5JdYUvHD2lOfvV66u"
            b"HPQltdr0JXzaxYWqi1vn6bkMOVtqH1y9sgt7sQPQyg5Aa/1ons/H7u+vVO1/IVVLvDQcFLMCS0adNkSBMoZyGYtcYQQVhjNuHQoe"
            b"I46UJtIjktYKicl6dfhKc2HVWfgJfNhEMaCFVRVq/oZZ2LbJ07IPZ+F9eSCSzsARW9rrLdBWP+Wsrs+XV8S2DfmaIFp78Be9HMTm"
            b"5UBykKz6snM7B5ZD4xJLLsFY92r8fJqNJ/AO/v+REC0sa9xro4b5Vn5vs5417itsQuVH88d5Pz4vcn9m436kMRsUE94ysMhZxYXT"
            b"MW0Bp8pYhZ23PmirhUIq0EA0TsxYFni8KhaT3R+luC3WslvwSnaLvVbeWlFmNyAUvNyyz+y8+Q7ooqhbiDtBlhwRf1RUqUVoiklV"
            b"HhKsmcmX6PwF7g4QBfUKkpIWaLU8hm8pCAPgTrxsG8tW50+nTYmPipO8hasacn5VKTxdE2Pn3+u2L1odk68m/dYwpnXD4tyjdPEp"
            b"rGxTED27Iez4g+zxJSko722Abvpk/XWR/i4Y24LfNkV/1m5d9weu7RigQpDBLTuGbopVG/V2dW4NU9iFFP94ufBm0NWo1X5Fk+ub"
            b"1GpJAEeN4spzEVM5EZpSF9M210wTirynJD712AYPgcdiNHDO4zWy+J3GyrxPQdhE6NYU1H0KPbYwrvsXF52Lh5D+G32Nu6j/QXd2"
            b"ay7KVx48dPT5KvlzQ7WJnX2KN2WMhheVo/ZGSiY67V/BOpfJ3WH3YESRQ5WL92yBFHO6gTXcNb62BGPnyjAG1tZg7+zXn+kO7ohY"
            b"WSFiZYWIlVU+kz0idsth7TO4wGR9BhX7h+zHZ4pXP7G3Hgb0MfUgSonWXokgHeXOM4WIss5pBIZ5RSjHgWtLrEFYM2uR1topZePX"
            b"nfyBpuJ3nKxukpVWYNa+1XoXe/pA8eRBX/ahkfZtsRN5+n1xVAP2y2+tv4xNG6V4yhA8Gp5PVAav6WWDjMqW11cyKuMx/6GMymNj"
            b"WkIxxM9QPkq/hVK1F5T1s91ZnNeY2MODkTfPefdqpsrVbK3m6Sr7HlrdeyLerWp3dGsX33o7blHlT/3XWfzHncWpYN4IJ60VCgMN"
            b"CiGlkKGJ8sVibWuUIYp4Y0LgzsQ6F3QgSsYFg4Gx/+ao7fPukHencvdGbQ8dGXEFlkLlrXftm9Gc+5GN5cwLYsWomIgMXDs9znUG"
            b"arGYfiXKw7l0jw0wlx3Q3LlzeY7dNQoEDNM5vFj4SnT5HcQlAM/K9BZyUqK+OFt7w8tXHivG3vCts/qe4zmtOA/8XBVerCfbGrEv"
            b"KPXTzUb9kEmgl+tFxZ2YrBcfU5r5Hbu9u1JIbiWIQLxXUniObdwwBGVAIcd0XD2U1Q4LiSVozUBwg7EGDQKHWOLI/+F+9ZJIwQbh"
            b"hY1wV9L6+YbibHXt6TTFpLeVabHqG65s7GXkNS29PQA9T88qiVgyUJRLaPwpKl0YtuIAvrRrS6/hS0/M9FqiUKvDc+Fs73eRMaRc"
            b"GMSvGpmZWaa/IXzhFlf1wjvHy8I8BDQh5DVnTOfKYXGvjNpL6N+8XGWOe7cdDc+UcK71v1YaOSv9r0udsYkyWP9ve0saXmos/jah"
            b"nzehqdQUYy+opAgEChhiXtaxynDIaUZJzNXgBPKCoxCwsVJLLQgOhsvAqfn16nlgwtNMH3N7YOwNTAh26UxkKuTYcsP2jhRqL2/i"
            b"BXk4+qjxtG0uJqtsTOiZa0mu9M7bGfNtvsjTvrIbbKI3Ui3Lf86pfsadbfD5afV+6ni3uazdKreK65VP5cOMS5+N/+YZ9/qVeca9"
            b"fmUOWKOTRy0S+deK53NWPFRJ7hy1JDVLwCPpk8yYTY7ohAFFDmHOgoypF3nONBXICulUcoIQwvy6o32HO9pdK7LN6KzlK5/jQvGW"
            b"6Vp+2/zu7Hz3rc/S33iKSQmaEAINf3iHLNVZLIakZEo2iMg11/gN6zdSQCJkA4ng3WT+rHoHozMq8qd3Glk2b0FveVg+LX6fEo5x"
            b"43lW8ztaC+E+5Y6FMVs6ozWE4+phpfwwS8e/zmifT8catNUoSOkpJCCwNRRZTVIXW1IaRNx+OcEImOBTdqacE3ASYYGQ5eqvEIK8"
            b"p974QbnIVqkBdlWHDj17c6rK5d7/rdkMdM2Ay4FFW0LwdnM9n+Hd1jW612Vfkk/mFhFclQuliLJWqG0lyUPZziXhhKrXA8cSRt63"
            b"XUdKpJ76YyXIRgiy0YFsZCA7FchzJNk82zQgaS0BSb+oALnr6/wqQH5Qw4FqqR0HkiwYODNAsUciVrvcxaeO+TQYDFQa75n3inMZ"
            b"JBgsmZMBK8Z/NRx+NRx+NRwmerzpk3zinzPXcJBL4odcwo3lkvghj/lcnVbnRpPn678aDt+g4UCtRMQKqoPxkjGugBlvMJcYrGJY"
            b"glPeOmMpk0jEApZIR40DHxyjnOk/asy2ajZ8odMw2P6WRmdl6a16/kZ6tR86wdUGf7K7T8cnBAPZ8uq2Rz6gih2tY6ivpw0LefR+"
            b"0ThoGtGGOS7+oDVTBBbhOS73DZjqqucJHWNsxdDho84nyngMsaNO9k8h5+De9Td9Vjl0RyY3qI1vMBvb0yNUia+1w6FVC5beGofN"
            b"LSPOYVY74PqUzcNHdvevbCGvd/G7+PJkL/9xQxxqHbIxoQVniaRYEe2w10gRQgIoAgEpJBgYz2O6QzI44pwHz13giATxx/VS0Sut"
            b"Rnal1XhD0Pe1EOKWcO6J1Lzsy+7siC4hjfUb2iUKxvce7HrIjMyQXbxRIQNvugpHd3Z6FavuLGvlJG+KQJZ7yU1UsmMeDhlFXnV+"
            b"5SgKkaAIBChULhD7pc+PK9HpVoEQgObXMVbAOaQAON7UbEzfkIeijYvdf5EWq3XQ98cF9PZ/GQe2be+rZ+xCF73mr53qDHPFXfx/"
            b"+LYSw58h3nivl1rBdr5XvPG6lxr3+Zrx7PdAicVEU8e5skZz64jW1nCFk/2uUdiDoEJIEaTx2Bhgln+W0fYMDHyNBH4OyH0Bh52k"
            b"t9OSpr6bZ2jhD8reDNlyo0UQXKpCUWEIYMqKoCUWGEfoitt2p+98ywWovFP6yc6iXh6+k/X5cowoUyVBXhDqSgx0G/en8Fv5DBKw"
            b"miitaku2j/or40VcaZZ1FmQjH3iuu9j9az2b+sfAtP8dZPbInlbEujVu2LljTkjKwMfalsctvFU67ts1RdhhBBY7SoJHlAvg0lit"
            b"wDr1PcXsV5wo4EWTlN3vG943LF8gr6aOjx3K/o72w31CHj/JFhh4g7vaZ2F9b3ijBcome6HZpZb3z4W0OjALu/jMMYHvP7AcnM4/"
            b"mEviXb+Mv2Ex2deb6dy5PNoYyNO3mJtH3HYD2TkhuBAHxT7cO2V6eo8Nsv8mGKZUNtfFqlIbBiNKKbZDDvbFUyfKh9YUVwIOVxYU"
            b"74ihVfl9jkioHqa0XzlX1CYWlXjEv+VSWZXaM0eKn/evnJTdTDqvLRMJq+CkMIYnM7WgOeOaxnpcaApOa+SkTn6XUnAc14v4X24N"
            b"WntZfo8F+oK6V3d+L/QnyIdJfesR1aBlOeMowAVytYb+wxWX4aCZvWoFr0r+Dgd8Juexs5uZuQQwOqzjYF+he5uPvUPDG33Lkie7"
            b"DyefMO0mQNKKbzdXwsxB2SBpZOf1H3p++8KwYxzJCmV2rCayBwXnQOCAaQN6y3iR7hehtrgv1fWlAHnGqasnbtDxr0dLIlhm9o0n"
            b"103pzld6BsWRzJuHR38E/gF5ytY+qPDhnhT338Sz4AiYQykBM2ogGM8o0V5ippmXXDqJMbUWIxRTslEMC+cRAhn/67n26u+2br9T"
            b"WN9Flt3bG8wUNkZn9qEBAlegst7XDS7V0m/rduSQfDsKBvzvsOUpb09w3bdeiQfnqJTvX9ut5zc/Em1jAzr0YnKAKO3kAllWU+2S"
            b"/Kp827/9mZRlyznrH3/0lTce/Tq3f65Nwgl4Y7CM2ZIGqRxxJhgRy10ZnAPKlaMInPdKIaUNUECcGYFxAB+3b/pX/fKL6penEcSm"
            b"aFnfxJZ4cvWyfXCNBND4lncpzouMjoekfpv18YIYN2nQFPZY6oRgKtFcSnKUFG4dMToZ0BGIlySX8ukhYXiqQ2pFiUmjIkfTnMAB"
            b"ph/SqGCRQ0UzGXxIrkjC709QabiqeHFXCc9tN+VS6lJedEoaDNqhCTGggKuHvEYE/4pdfrr+pcRgIAEkJwzFx8oary22zlkwGlHn"
            b"aUzZzAotsXSBSiY0SCXACmb1d8jGfdECHrdorwX+V32VYDGtbm8Da4cpJCy4xKMcDwxQtjVN7kb1PdW3m5LO8HYz20WIV+NSOm+Q"
            b"F85YeoNN9nfvM5BVlzvdV7poIncixgYkqW2V+4NkJmNsYsHzUWJ3ZdNp4oN+A2GM0kfoto2PcaTG+hkczyq4RiUKJ3fExuFqdKjE"
            b"X+A1GrGe6uGRXSd9h49qcv4EJ+MbwMHfBs3gnAenObYGk6S1Q7wCR7wAFktoigFb6bnyJj7goCUYShJMQyFqEaXqZ3vErxK2vEzY"
            b"+MOcjYodVrMx6FNtuF41/vA5bsGvdJeNF1sDk1eCa6PCvGhV49lV+CgazwaBsxea8Y3I6qVm/H7qGpmMXknGs+qQnKQnh4yK8bW0"
            b"2z3F+OqIW4rx11c0U4z/Vm7HSjD+isHx8Sr6/Mc3aug/nlb3pST+XUW0EMrFbxNGJmjEsMbGMQtBccalYIRIawlYT7m0VFudKHUY"
            b"K8coosrDH5XErzL4h+rtoVUx2BZNMvvDtC63UpfvAYddWlf2xbdJURI2IeKTgsH3jkejA5Tz4OFEVMeLqY372EaZ7gjoUcPP9D+H"
            b"gaHgZa7XMa/FTA4iXWqxkEM7hIXuK1HnEZoj3jQ52lQsH7l58lxSnqxiUjtpEFxLU7bPWAk9EmL77IRj8AaPwStEM98Ei9le2G7P"
            b"ijjlfYeNPyVhlrQIdVqE70+LQypMsgw8EIx0rGRxUD7WsIgr5bGzNJa5PBiHVdpcYuK1Ysom3foQ4lfHqWA/K9twp5NwDUJGl5L1"
            b"XcmXq0jM2j14aywxChqM8u5kr+2QzAnjiAW1DWrbc34pGbbdgRghJBoU32tqhMyIAklpE8Wrj2p7T1rQBEo0CvJ9qzPfpMqZjx2n"
            b"PLza0HC3OaxgK9D5QctDp72vGfPUWqDWTOOpovszpZoN+1u5ZODKrRIfpV5t3043KNhOopCdUg1USjU3PIor9sUfL8J+oZPwrYLq"
            b"wwBLaIM88hIhoyXVyS3DcmwolpJJnsAABDkqKVcISRFXOslC4Ipq54yz+AflEX5cY/0+HGvaJx3+6Duryfp+lhaY5NqJaArCmoyS"
            b"9i7mzi/p5AuPMkzsnUZ5MadBm+SCPO/lyvyYZJhX5YspdjTFUpB9Znx5DC8OsQQ5cVkuM7F5+/aQdO8OzKaZpUNLCNpWT1nt/SeE"
            b"mvRRlegvwbUKavwJtIDmaztlFOQBBqCNS1H7mJ6GRdWTjWnM6x06n0C2OnHcqXvmnwrZqluhR4W48jD6OmTr2/QUhItbae5jktZM"
            b"mMCFNQZjy1jQMn6LkDAUKCiDiQBNtHcMtEXUx324ZcBe5Olplv6QhM1/1ijd0+1h3NtOPm7uUTPKqqD7C3Ps7L3NNVtQiSS4bK7F"
            b"zruliwPIJv4gdxpAIQ7QIbuliINId7T+Ok+iDPsVfNYebUAR8TwpTEpUY2QXFxjvIAXGBXGfyqEF2CF9P+L/0ZNz3K5v6aUf8hPG"
            b"gwbCmSlnnF9YwgFgCQeYayiMWfIvEJf5wl563xujKvVBtbte7bHrsvTYS0sklXJWxqRFgmbeJ4ifogKcc0RZ44LnUkiGCVBpwVjO"
            b"Yr1gjGTaK2n+deGYYQA9E45BL4VjWqOAnDTwrt6CTnh9r4YdYximncb1GHlXOYbhvQO4UndBDwVu8okS0JQLfnovbNOeUYZ7KhjT"
            b"k7XSqTLHSxJef4iTBJyNH1JYIxbzwMHykVT3Siem6MZCrXHYMHafcnlPH97Wm/ce//YPASgVGNIsvX0nDGloFUoUDAFhtJE6bh8w"
            b"AkalN7F6E45SR4zkmFARpE2M26CtDZ5hETflXJOYDH68VfgSdHRRlD3l1d8SNF1ibm5giMQdOFIHCUpOuxS13KDXXptTJUW6E0Vf"
            b"GOxuFrzxBz8/i5V6ACpxhOAqmB+D9F7cNYW9WYE9cHeUSwEAuRQA2KbAVaGGuxQ2JqdG0m/+cKvG/koAzw/CdPb8BMgyLqnF1Ars"
            b"qDQCW22BMB4XN4eEIxqnQS7SKMTyy5jgHWHGU0fjxf1l1CD52lXxTWrQQpdU9iVVLU2NiirOB6WpUdW/GllEN9WkRqeYha7zE6Hp"
            b"uSz2vlM+Ofl4ZpGYYElFchoTIu/IVG/iVqnTl0Di9RvSFW0qB+bLa1lDT7X/MMdA6SOE5EqxCi70quFSrxou9Kob/mW1ea0edgTN"
            b"Gj9J/3tC0SuNwMbxdlenWjne7rvh+FvOdIkZrehbOUUSWJKh8k5KAvGrh7X2IjDsZeIQOUa9ttoxrpWgUmAaAnbIeAIhWC3kd6DT"
            b"v1WG+oNS/1eet3fQ6CsZkbsiMJ8xvh30YtbGt/0V3DOPHSDy+ULyR8anu/sJz7+rrw9ZLzbsyaeWuSdJdNJwXFvmLnyJ83XnaIap"
            b"aHbpC45p/nRKcONjez+tsyTO/dDNpXjA1v615+6b0xPp0z5jF+a1rJih19611TN2WNeyQxqbnRD45mFlXNvM1k/j2o9obf15/gTj"
            b"1KdRVfl554IJYl6CQCCMVCxpZHuwjiGDqVJOOUGoTfMh7YOKecp4I63gDltrmTJIGfb3rQk3E+5fsHQ8YN8/XBf+sCSMN7zVhlAS"
            b"B4CrX/2eLjbPs3u6iBLKCKa8g0zNRQ9K5JtrwVNN2pW4CruY1h+eX7w2/Wpe47UhGG/6t3TyaMz1Z6f3RFTtVf5v/v8v8z/H3GCE"
            b"gzXcIB+wcVxyIbihwHmQEIBlV0bPifYqxA2nk4wSENiT4P9tbfGbSt9fNdAdFoxbkt03ekpjbial17vdR3/GWxLkD3XNnyuW32se"
            b"7RuSbHTLoe7V7Bo408vJwUVUHBNYfCqt0mOK21at95TFdzLrk0x+jdeXWzF+cl05vYvX357T+sWKB0DwwrFxh8QWuP/kyUtMwq/W"
            b"+FNCq+TUce+Ed8FaRVjM0SFmZEmUMBpjH8AEJETQEF92znvrKVXcOcUFtn+Hb+MHa+17FpD3fBsfGzJuBlypb9yNEGfX8NHGT42F"
            b"5c07NxuIr7k3ps+BxI1gk2ppKw/W1/opHDJhQFTHycr1DHrRB6FO28e3SKZEyiSO+NjG8UyXzTO8J7mzM15L0/K9w3K0xjn9RivH"
            b"vWnya+X4yca41IIFzL0AroXmgiFNuQiC6YAM8yRWyI5YysExJZFCXnLjKCPIWC7Q/7RswOvkfUNZYJTbkiOh8n3txBVJ9KZ+y1zc"
            b"6hA2mIpbVT6PE8GCG+vW0m/yioUxV1wnJ4GBlg9ETQQL+mvKgln5gMP3saGdjff9Ze/Hp6wFvKuznJn5hsxszsNQJ2Wo5pWrTsiK"
            b"o9ANMEex2V+ZgO+RCZAKlBTJgNeoANIQBFQq560LmKJALQ0hSKy5ZAZJkUixlKDU58aWM/ubtD/jgX43235oFbgvuvUspQ4WQbLi"
            b"dfW1MzmLZ1pQy1XTas512LJp2rQSULwjBS8YHzmySPJyQhrm7wJaIrbIxjJ97l30Rbf055wJaBwlTyje/BVcSdLiRp52/coTudp7"
            b"XpS/mfmtzBwQZhYc8olT5mMWJhxxjySyXINGhiqhPcaCY5ac0bglnvjAGBLS039CBfHDEocTqYNRZPYy5V27Key8Yhjcfa5qzUGJ"
            b"pnGP2JsVPddj6R+BepMDkuZmuHIz396azigXvDB0WW4II9qamnfM4fR6vnngSOHzcmUlGDwiA3Ns8lLo8dAd1lNsQW9mVk4faWTh"
            b"KhniRu9q/co8gV6/gjcFrPPRnkb7Ry1U+lfg8JP9YB2EiJkUEA8CrHTJRsEygWigic6BNBUC4lMvMGjJ40uUCEJlsPEY9et+/tj9"
            b"vE9YldD095uas5yp0xjvjkt5JomlQLLpDJ5WCC8gHnPKWzpLem9VkAuiprs9NUdP50gRXzI7T7+Jp27no0gg/J9shLdxi4L+gtt5"
            b"a0bz63b+HpHNEQ2OESux4UqzZCqjZMxeXhrjCQ/eSgMkeyIII4Jj1BBKKTGMaQi/RJFnRJG7TjN3+CSPiSLVyKyWk55eaT3j2oAE"
            b"h0wg3ia/HUH5nqe5OOW5KZOVPTmt9LnxcPr1TIxcjgnLTCz+sakibtYcdLzZcFD6087X9zXOCBCKQQj8bD5Wg4b7Z3gH+J4AY35O"
            b"uVJYhU3onsoCMj7mae2zqxkZPV6njydkv+SRd2ZkPlAmsDeUKutjOcmCNxA4kvGPAAfMnQTi47bfUgkQLI5bexWTsmagFRU/qOdF"
            b"/wBM8WoX3yp13RGPmaMCnup5TVeI55OkiUIXXbdVR4WuLmGPh1wrhg2fw0zP6yD5rUxyr/W8aqjzRPr1Us+LztwWpnpe34cAXil5"
            b"saWS1zU2eK3kVas27E2Bxmq301HkJ0D4j5qGLZVrvqEx8G0yXgppK4lBGKSimEjPYq0sA+fESCeTOLZQXhGw0sZa2TgRC2wCChB1"
            b"yID/hqq5Hsk8QfDiVx2ACw2b1TCJ7OOk3sNrUteeb3HMVGKAZPuZWFvUiktCM37RpC3yXd2lp15jKot7yOzOWX4MPU5n2qTGKB5q"
            b"TjmcVbISSQgWzTZk1J58vV/JJ8nkl/eka9JbPBMynMnF7hKHUAsUVlEzuVhYysXC2edcPdxK1z9dvmYGhr0pXvNBcQjFEA3xW6yR"
            b"JZoTrwORFpgg4BnCOqGvEDbxKyyxF4YyHggR2npDDMUO/ihv7ytj7wsZ2Rcasld9zamxdt/YXPhv91Sv2d5zYKWNm+GVnk0vsJAu"
            b"KAVRpVprljP6Pb/xpeqi6DXJ+vbrheP3vP2K98JdTi9YbU3YRiP7oXxr3lHc36Cv7VBWnU657HReSeKcNiitNUo9BPrzTbWb6q38"
            b"AvGPWGcPrU7FLVgsjSXIAgKOgRvqnPbOUG3AgI2JzkMIykiW3NeUA2659MQKGdwPFW0nswpekavoFbnqtZfHZhDSFSOLguk1pKkr"
            b"RBa1Hu1JUGNLcvWmULb+8Ye4sCmArTkgMZKtI+7WAKruNYUkrk+dSI8e5P+zd6UJzqM49EL9A7FzHNb7H2FYvAAGx85SX1VNTfek"
            b"nYpsY8cRQnp6j/TGK+0XQN1Kq9afi2jt07o1G8bX1ePTjUZ2abzt9smXloM3ydk+lew9Bq11Mkq3jqGirt2fQOxrpdXFJssyoKe7"
            b"q3Jr9S24/1Drb/WPu+BUTQbWUN7guoGKLvyuO4frZYh/pztV60rRb99H1UqzluftiPCX+0P+2T6q0+ylkpQKZ0nqnLJaeWZjeISt"
            b"19LGBXJ0wVxgLbGhifyVBoGjR6acaed9orn+rTQHH+Aw4DvZAAZ+IalY7NLlUEUoHQvR9QO613YgG8jpkV6sA0NV0ccI3C/XrCkj"
            b"gh/gU3gkj71JhfODmutkVHS5G+m16VQ7uyuwjCm9VjDa87EV43IeTsWquMMqVsj5MMsONNV0BBfrnkOSyGKRvrCXeHFuYaNyIaqm"
            b"voG9bnXeucVp1azVvDkPnTd/XnXGNhPBmEtyrNL97VKl/ze0CMo4yi1SOkbjjlmmDafUaqYCdUwiEaQy1jMRwxcdZxUhrFTYE0+0"
            b"NvE/f0Kyp0D/ywWzOyD+N6rT3tKDXYVsKQx6w7pcxpE54YyGJx2SfFT4Nt3ZfJEY06oxbLvRM7HZcgV5J6BsBXuokxOlIRXTZwXB"
            b"b/PgjGRg2VSLYWPBqVu/qvrY+Gg7r81eAZuxmtf8N38dB2/tOFABC4KI414QnzjQU9WLY56oz5PWGLYixfouPnueKwWOmSRH5gAb"
            b"TIA8A05gJ/mVSXLlYyCyA6pq5XHJ2QvWQwzSeSleqEDYSQy6JH5LHUoea/d3AQddvX45Zlurf0CEUxXnu8p8yiAgkBVsANbcyX1G"
            b"9XykbLBr0+TSH3Q3c9GmQQr2776jWIb/dmma/OP5YMgLS3ZjxblClefAVfmqzVLkZAZvaIEreFf5Gir6gwreVWu/Ht4dwtt/Dsl6"
            b"quS/e70R6Oqthf3NnTFnEeEhIfqVNwoTYgWR3jqriLOSAwo+UGoI0UjTIEwMQz21VBgA9wF3Nof4H/D9B7atB6ozT5FpVZX2E+Lw"
            b"DUxE9tTl4Fid6xKrh2nX/fEguakfiUr/RT7uwyqsWFfZf8dNA/msWWYEky3WElWi4pA9GEaaZEi5u5Bo8dyfKqtQbksJt048W2Xs"
            b"GFV7c5Ya5haySY5JVw6BaugH5vYjf8BNrED8Gu9jpY6QADb1o2xaa2MLxwCtamU7By8+8CRWOKlOera2/DbwgZLrfeAx52xah46C"
            b"M2jB50BSIXpSywmzyilISV5EHUSvKqOvDVo5jHH0p4gziCEkkECUEJJKYyghwvzBDJ6FGfy4Kv6glYotlax+x3+IpsiDy3CKF7AG"
            b"8Tu+rw029n9jrMG/0Ab7wxr0S2QtJA40OEWZAGec1JISRY1lUkan6OPyOWipUAgOgJm4brY4LqKtoUkOVv6Kpvypwle7ZnuKHuWl"
            b"9v7LDfnZIIW+gFEt9NXjMm827i9w+NwCVdzXoQTXBmposUwamqhXHCPHtoa1YQGqdfSxa4Etx0uvbO/ZbQpRTeUqmUG2B3ZafCoW"
            b"ORD9ivQjrhwj7mQSceXwcOXw5q7wzH3iQ/px7cXv3eVVkta/rvzbXfmJM9BJg7hW1mNPBcPYK40cMckFOK6McfGnqy03hDPhPcIa"
            b"AmICYcPNtyPRxu9Vkn0Mir+26B8gwfal9V0w2NWl99g1r/Ctc5cLla7BATpWXHlDKTg87ojxaug9xzx/MG9OOmH4O/YxSbWvzXdq"
            b"KfmfajnXZ+RSNxfyBN9Keo5JpJ5jOjnTl4QhB8p8q9j9xB6AF5fwn3O1yqL4K4O0esc6hqki8QFQpo0SViigGGT8iHDPpAOClI27"
            b"OCe91Zwz/EWQ2T4Hep4AvYDbOqQ/x0z9N5kEZhzQYuSHpnjdnhFw7K96Z5UtxJqZbYn7Gx9F0riFJE2U2YV3RwoS2BUe+57XluoP"
            b"qmxkz9kXT7pQUUf/JxYOFFhqZqNblFRtivZkyTR05gchyHTUbPusn0zfx73A9Kz63SH9a5WXxQnW26Uujqu6OK5C1loT7KAMUykL"
            b"3OkM/SaK3w3udYZ2anGvH3SXp7jXGHEG5wgYSVh8kAnXOvlH67z0UvOQftQ2hqqGWeF4QjNREqjmyBNLNPmNko9XobHXIa+X9cWv"
            b"yy1epl+9p+x1B0/1BC62SUxsHpIMoar5JpRMAqOw5RBOabLLVWaEMcatKMMJuhgXPWCa5DNGQKUjIDaNp5gn0OoV9fNi9ySw9R6V"
            b"6oHw9CGV6jMkqyPa1DXqPaNSHe37h1/9cvyq9hC4ik6eYR2kNQpZEgjCTmAmiBHYsUCpdpRiLZUAGRzx2kgR/9XzbO8/wa9Oc8N0"
            b"XulCZ5WuAbdUzzzd004fQK3buS/PO3KJi/lqoCZpzXiaZCVhWZEvLmcD4nenlgsrywKqqu3FqKp3tcZGN/TEaGLpK1HJKsXWvVhO"
            b"8cvdENJQsyVFQq43ffH6nfBCtmjKW3nJcYNNVUCcmvgtH8vzgmLvDKjfrZ0BC0CqfQfFdBMNaN/BqTAXNPApWHm0hu92yw/7VlaX"
            b"wFBfAqOvlsBwflrTWY7+8gMlsKCFS0wojkU/hzAoE/0exnGqV5YQ5CyHuCm88NSBw8ThII1GOlBscfhMmvYztTE4q42xYW3sGebB"
            b"UeA8XIsfc7R3ErAdCrQs1nnPvdKlRIalvQ4SdiaZezHrKnf2wJ1Nqk78wpBGq5NErLusjvcqDzeNJS5JNjQCX/IgU+qpZFu0t4A1"
            b"41rreuPRZfOF5qqhGryRleCSEkYZv9WVKxvXOH5HG9zpznIw6+hdJGEIXl159U7uIuXN5ibYMujGWvCsm0zit0G0XkjtNlW0eWr3"
            b"EdL1Y6ldg7wzilFjueYYgtIcISWQQSCYtBQggOIWS46DEkLF8XEnaBDCc4f9FxG/vsUxn3jlBxRWaqCIMqRmvURhtTiA02rdBZ0B"
            b"cRj6gMKK1v6tJZtKZphtw1ubZElFwXofwp8OWJx9x2K1HPZFFqu7SH64V9Eas1jJKWR16U+tWKxwBUUds1jNmlGrzQWl+hPd3Bc6"
            b"s9WBAXjEqdWEGKPjHMiNtqDjg4G8oSI6M4MC4gEZjAMVkkDQ3lnivI/O7oczV1+jm75MOEDKqotnplH1iOr5Gj31c7CAO7KsA43q"
            b"VOSqGi/WtVlf5YI1iiSoFSNYUGqH7PSd5Gw5KBQYPt76FwCdElwX20UmhZQYct2p3N0O/Jps+KsE1fnhuEvmApV3hAqAChWZCzRk"
            b"LjXlP/0YlcvrIP4/Juqjm5XIe2MBe8+F09Km543ygC21cc1BNILAnEZYIOkV5sCAcwfGymAdiO8OAXhOAmBYip9TZ8nHvFid87yI"
            b"TJhKAMyK7NmwuHuxl2dQy77/hO8dBrJD9pcbGIjM+Z/GSRFlNXJh1tOU7Qht9K/lBA4wkr6+CweI6yZ6J9W5dHs2rZ8dyVXV+FnR"
            b"n/KGEKt9dyJ73eilVoKq1WYnqFqLqP5hA97rR71WyoroN732wrlAjSPEI+koskwzH1fajNiQpVY90+l3InAcpCEOoW9EGY0e8RLO"
            b"SQlbpOrCDlIE3sov+yAIlbJHmRxkaT4SK8EKOVCrZhOMFG+5+fqWJ15sSHREnU85hJ3ZphaWHpldWc6mQ6TPiVw712eh5mQSafKT"
            b"6SC5xo5KdFnPI21uMX6evCGB6ttpLUgudmxk0fe4ACF1yuJb1KljcJOcgpuKMyP1UpvgLhU5ikc7iqfj5luSil9B/DfnjP4gvd/R"
            b"fTmlpFAyBoJGMG2DoVhFNyVQfMpM3PLKKpJ08qxSTFDHlXCCxiffA+XqGrTppFx9OytIHlWsT8rVL2YFYeVul+dZQahEiOQ5lx45"
            b"FqKP4KJ+7HVasKHCG/WOXyPqOFLb86qzSx1j4gO3fd0ytFaX7xT2D6nBp8rU9AtI7kfpwZdI7h+mB79JEXoE47legn6n0wrR+Qhj"
            b"ODNcYeoQS+ULwR3hqcJBtYjroejYQIpAhFKCJcpegXBcyVLNfiMe8wWl+e+H20wHK3ZNIm/ril/g4Idi88dQnF8DmSyG+ftjHYkI"
            b"sIlWfTGMT07coLTiVNluEYz2obl+GzdW7tUlTTq8seVLKMZPgjPL83RnRU1rBbvqDZwSksKiwkQrEaYOcjQgI4UNXw8b5r5ZQleb"
            b"FSMp7HEnbIJ5V9RK/wCbHwJsWhSc11oEBgKU0Qaj+C9wpbBEgsVFOENeUyIJ0oA1InEhrgA7KoQMHH1CXm+mrffC7HE+dXT1jfiA"
            b"yW3hvLdGHlJ42Y6hVY6t+NkTe0jGDMrTUtA/j/e5FIyyFV600TSx7dADD0hLv3zFSML7ftJ0jlpMegxYx9kK96L3j0izh9NsOkrm"
            b"6BP7yIajzxepCkPex0Hv74S23wO9f18w+30Su69Rp1s9GgQvnFbKMcc4GMuBKBSDi0QZ7mMILAWWxrvk3xw3gJTTQTkrmDAau98H"
            b"57lS2Lla2xhiHUUfiKL/hFBt7TmzZXSYyAnAUlwBdZ6zFo2DsmvwnVXchGzaJuPeT1os4gs/XirpWnWEKoZPonjkPXa5cSM5+29Z"
            b"OG/+qOadq1fm+/YZPUhF7jHe/APx3Fih28R2hJ0woLx2JiSChBiFURbXUUbEpbsW2nOLQREdnRuA00xa7gzDUhP9hTrHI3a4lsDj"
            b"lAmOPySDK8iBK3RG4hHKhww4ino15Pp6ntU5VitppbpIhlRuRI98ecCHBGsq5Lp0cX/x8UjZ1UvaEdEPxr8zNjc36vA1r13rdCVn"
            b"IpjTRt94o3NalEA6sudsP9E2Pn5/M23juwR0d1fYM4VjmCocn62taQUAog0waOWdqxsfqzp1tYLeetjhLZIdb2Wmq1HfG5N/t1Je"
            b"CPtHK+XSsEMuc9Z9jLvT4rgSZtpIT10MLaPHdYoYboXHwRqFlfTYAmIYAmPeiYBc/AeBSwTK3v24zOmNPOb7mt6vaibd6iO/mN+9"
            b"LRw1he4MmfNghdbEV9ZQLg3NR/0/eMSzUka7pD6rRqZVTX7xtoP7kgdC8oiIOgh8jBfd6ZaX06T0Kd0lOza2r+FuYknMxuMTULwB"
            b"D4zPk8dUjJ+Va7rLxTxnXJZVfUpWka9sIJq4+WSm5bGu5w/aHRc069v9/5rb/1WuNP5wjBUCPPFUBW2loNHncyeFlkRqY5LKvVCM"
            b"eiOQxUIDQ8GB4Mxq6X42zv6qO51RIh3B8wN90t0lLt98o09aJEexRIPQtTxfHX4os8ZdQ/KX46YwFMhaFUp3qdEdaU9wSfq1HC69"
            b"yhr+Ts6PXDDzoyam8eCnerCk11QtIwFGaa23NxhRS0yytHRWhKdNT9NxJXbkPX0alX+bJZrXNNH8gj+fE53OYFWNw+58N+23Sv3r"
            b"ku/+A+Y/ASi1RLKAPYoPnTRWKascV5IHIbnkSbbEWU4xYCFcai82zsnElhafIKyx4X8h+pUQ/dNhcjbKkwchvMm7jgPFi9PRRert"
            b"y2E34LXRABcC6oWRH69L5KHKbDaOgXBz43gbqfdhMF924InW9wrV9YjYcHgNyagcNYbYmK2cqnP2gmIWb96TITm/X3SrmgG6lMmx"
            b"Swqm7NYw7caqC2xrQH6WZqF9Y8Af19Q/D8eF9CIIZmiMtWVQMfjWEix1PnhkiaZUCQBBuRE6gAIdpPCcImfBxf9+wu0fFALOuaAu"
            b"JLqH9P7LDJBe5RG2RWg3VxQsDlENpAovV9BbqzJhAFdINhCB+Li0xnydgwiWpNMrlTl91x46HTDNRyBguU9rKgAfBlIOmZ0tVhwd"
            b"kAjlaaTnKaRhmrocr8wcDCHZHzofpbbPRnm6IRyRChRRue7+UhfT+LqG/Q0vy2CXHeu3EnPPGGKKSbmJWFFGamIWVZ2IdDco25Yh"
            b"RT+OoOHSHqtsZ7N4Uxu/fxdifLOzdhamn+lfz9IuZ4wGe7fsCmAbaWB3m1tuRm6zxmO27e/CiNVE9Sv3yiyqr2YAWqfk6ZcAmK9P"
            b"Ap4zQhHRTMlUL9WaECGsQ9RRIsEFL7FBnroY9HPMBCaOgcPCgjJ03pT7T6S1Vu8wr6z+o7Lqoscldlgx72SreprCQZ3veSmuYXE1"
            b"DSdLU22It8q9NkdsiwZsjbqH94RVqxaJalDiWms4anYVywKb2zpfaKVwu82HzS1bdB2RaBF7ahelbdv/kFhxKQVbs+1SlYUPA+Pl"
            b"wiEpzm2zOV31I4d75eOXHV4qr+bY4E7uRla84LLhCJ9VUVeFbFopZO+ZG1k1GNf7jNSxK2LwarNKv3+7+urMydcRfOPkX6uvfkxC"
            b"2yHgWnhQmmKDQSAtXcLvxbWvAWSlwGCMAEsURol1IXCOgwxICuaEfEobcWmvfb+0zaxP+H2iNgfC2P2cA/oBdcQTp+l8ASHXbbxo"
            b"IOXasTPgtbG2I0C8rI6z+KIKLtzIGo69PCrzhgDeLkYq89aDizzf8RVH3K4xxicZTDlDRkhVYDYJeUzpWSM0g2KyNRPfY1Og9+qX"
            b"YyTyKgYGlRhYm+8+IpHPPtnrjyPJr0f9G9+EJWFZx7aglItsCG8khHUQ41XvGZUaUaJCiA8VEBdQnDJF/Ag0CXEUQWAXAtVeC4KJ"
            b"kETrQKODfL6FeJEqfQG6dx5gPs5lHMBcI1lSsuVHTjO59dXcCx6nuelh99dBuwVXYdohN3+UYumX/oO0d5JiGUka8ps9jRO9w1ti"
            b"hwelw6WMJ8goU9xmj4pqzmuwPHwnWzDPCcyAy+dKW2OA9Ixx68hYuP6VVls/OGisMgNLeuxOyPihep+LLhN7xJkCq51URui02ECW"
            b"odS9zAnEzxwFbImw2IEwWjuLBDWUW4d/XL3vEx3IzYp+7ffoVLhvyMbwlcJmbcKoaQG79T3NsGJYHHQv2NKtYHNLcwZ+EMLxUK2l"
            b"E9AWi2l0wtAtlNGSKLusEd6nBjL8rQ5E1zJea4n3WikXXeA6CFn5MmdhAHysH3TXl2zS44KXfpdBWry5ujyCbMxA7VH0+F6Ue1YM"
            b"n+5bfkuDHa3e0c5qHNY+Vevr/vanKvMtK33Rl2uaKHZ0ipU5UYZZ642QlptEviidljTG09hx4YNLnDzRhMZJQiE1b+n7VjkCdEok"
            b"xs64xA6BcnHucg/u1DGhmzKgDeFrx6qbnWJhblBbS1vhf22dJ0aNU6xLi+jKIrsDbDC8RttM9RwQbIQhbHsI1xm2qeCl6TM+/Xyn"
            b"blArp8SxCecIYESHW5wPlUN9SRo6iIF3T7c5mz1JLcbpbQDcwG+y/2SFpZCdNvhc53vW/NelTge63/Tb84m16QC2z7VXGMXemQ4Q"
            b"jHtGqbQaMUatI1KTGORiERKmLWCsPRUhKOkUQ9oSR40XQSsisUTwiXj2xfZkfCYDw0cyMPNi1I2mOnTMlz7oqOtbA5+hDRrH4iP1"
            b"7gHu7ahfWBb/h769YhFfYE/+jgy3Jrz4ShveneEVF7N8exQeUO60pqrcmoNs9yGrk4a5yXZLqBMHPZKxEL5WCYNPcuQsxDgVaoA3"
            b"/LA1nyJvukIqbh2ouXVkk3SAppdETrhleU0uWzXwtdCzbx9oTkVeGumtRuTlk+HkaQjpCdNcY2+pU6mghBFiBJEAYC2TwfLUZy2x"
            b"F3HBaSzWwTorkFExhIxLOfWlGlz3OjgeiW09mUy4KpgwTDoMyycDBDFUy9cZdjZdDlVLU0YdXl4jm3mJovuuEO0yMzEi+MHr4xHZ"
            b"JMnlbFz3ebCq2xsPynLlXqTXljr95J7AMqT0KuoexLOhFePUns0oqAoXXY9xKmSrlhEmgdoHKjjFJH1fW8rhrshNonS7lzWeE6LJ"
            b"QjLOaU0yzncowZwU7TwbvaeHdz60Rt9hrITTcfa2kLTvTq3xRCqixOSjVMQn6TgupyI8M8I7B/FFuRiRxxnFMaKYCVozTSmmCAeG"
            b"tQxMOGy4NiEYQQ2mTBCMvpCmg36DNPWkKaUj4DhK0V4Msm/TdAynsHMCjePEWOXFZanw1bJhx3nvWOJsW/+6BUs6au5cjI9MPn51"
            b"9Iaqo+/Sy/ZLeF6Y5OtxyR1X1iM4uiTK5hJgaJwPnlk+hnwdA0HICV/HDfqj+FDdw0SMmTrYlKljxZjxGmTGd5RZNQM0b+r0x46L"
            b"WAHFfUrkjViyf8fo9mKE/zGiDi8ToVuMGClxjBmihLHEBvASHA0clCUccS6ZQAIMCdRzzKwmlBugcybLf9gO8gqJMX97CfEACv4h"
            b"JMbigV4c2yXjDgjpRyTGx+zxFRLj/rv5IhLjjZSY5oYOtDhiMadbKVYEv9b/8cdk/NcM8kXBueIWWWkNwSwYZ7kQTmrqNQVhEHPM"
            b"E6TABxFoYufwTIrEpGeFQ15Q9ZuD86vatRfi5ctxfid6Pmto29Ixp9RP+Sg5JSM3VMesmXwmFHxcsjQg540w7xj1cllVHfje4zFt"
            b"fs9mufWiTqzPeJQPifUT+uN0/Vtqnary+zvLYiUjQp+Nvfl9ijxcBd5XBEJmSA9aVSlp1xWOzyjyOtGQN1Lk/YXdh7BbGY2ARKcr"
            b"k2oITmQxHMUAO0bjGgnviIxPGVNUOAuK4+hsMSNSc0+4Mu6Bzx0iM2a4jFNQxhyRIeagDHjUu8Eet28c0Blb8nkock5ogW+UaJmV"
            b"526Bxs2lh3GlnvmoP4PPUg8X2yAyunmhwJA1E9w4L5EbxBN9U62zqYbJ8GSRph8ipGxN23VRvDfZhCIh17lg+Ro7YctSxkOK7hPQ"
            b"sbKaP3+yV+N+amJGNwdLxXHXsqQbcXNJGtdymdV+vI50edPqVgWsTWPbkXju27RuFCHLgVN8oXljRV+gygVChd+YoTjq2uKG1ggC"
            b"rImTKhgXAvKBCdDR2zkenJZCh0B0/CF5S6j0PABywTEdpOSIMMHsHxjtNhjtCsgr7Z8wXmQJDhdtni2auw/xIiobUKWOKDQ+yhZc"
            b"RbjJrayIjjyah2C72MWXpmsOJubZLrvQK+i5dG3Z8EloGsE3YsNHiF5aIXppgwg+1tNgikybI3yvLc//kGk1Mi0YZjlVLHXfeioU"
            b"xOWyCySuqoVEIYAGIkKM5zBKDtDaYIyXXhigziCvfh6z2j/ThBv1mG3rPz5QLyd8ocfZLrmjpDmWoTYAxg0msktNZ5e5a67yPF+m"
            b"Y0smic0e9b2+R4K0ZFQu/DIXcSba4Vepm4vZxl38PMHOV4sZPS1Z1L3rey12/3ulr+Kv1+Jb9FoEI6TRVlhvQ3w4jVHIO+e4lA7Z"
            b"gB1Q5lmMZS1TIJ1UiijNA2GGhvjg678c6vXk43W8xJUE6cVs60KPgNfuu6ZRD3WJiHw83irc1b16qE/QJusjQQ06GPOl607wUcMb"
            b"6rIN6/yyM8+XVC2d4CAS37worDd9iD0YC10MG5Kcgd2MGOcWNf0NkNosmSqnyVS5IBqg4sPZPxknU+WGYJCbw571OHfsZ3/J1Hcn"
            b"U4NDHjTHWGkvOQk8GIk0mOA8AIv+NoToeDUA0Vgw76wXmvgYYGKtwJO/zuYr8epFSuB7/MX36ZEnVA2kZ2sokfBFWt8kvScrW9Fg"
            b"s3rS4wG+um+OGUku02UWZEAUHu/VdBWnERVbhkGww+gGu+QG8dxQzlYWjQ3AdugRz/IqKNfbCD920bTmZQjF8smoG3MM9DaiYccX"
            b"N5iFBWi2u2lepzJILRJFcKMfVbWv0FEvS79JchzBd4ADp9Pm6CoQX9jr/0Lxrw7FdYw0DEGgHOWU2CTP7LgjcRawlmtliHCCEO0D"
            b"BM6QQSaVIjCmBDMjgvd/pbVjhBx/RGxHS41sZuFxGwqyJTYWdZkKfU2dbsCjVpOb7enpl3nUigsWFHV1u8W688PJrqyOzst3mcmy"
            b"lO+eK7m9hx5txgx8Rpw2765uwb8jojT4q7FdrbFpxJAEwQl3THutKGWWI8OIDUhKroFrGqgMSZo+BsnCYYcU9YoYhyAQ/wGlpTrc"
            b"Org5eVpJw6fFtHklrSPs+Y/8Vz09x15askZMuBIBKljb9nc6lrjHHapWLpw0SPGapocc2HpFsVm152eciBf68vIhMqG8VFBXu+gA"
            b"9NuxU5LqZk55Kfu7OmakpBPSuDyqZPtktQzfF3E+5lfPPRpUaQGo5J2v0DqOth4xm3+Palm9wpc74dSVWtn7BJw1EoFTjrSNTzr1"
            b"VDAAzxCyFoW4mJdGOJuYa7gLhFjNKLfSCR//r0z0ae6ZxCk7cWVzP8YmfgoeuSp65q1my9t9cUuGSvJ5DdjJpbOF3vFEfZ3g1T+V"
            b"2o9oxeM6h5etGlcmdgrHll8rGRW1h46Rodql6XVdiRmWzlVYKcaPttkmFfqgYAp2hnBS3dyxuyx2tOvFwqXhSy7lrO54jfNK55TP"
            b"l/qfoPUaSfjAQJznUUH/QRFqCPms17Pfxn01YRddOlNecmDvSUSuDkx5w4DGh1QABu2CDphaS4nj3GjqjNbGSQmYOkSS5GWMzYwS"
            b"JLo7LEE+z0o7gijdSzue5xzf1FJ1gPEcGokGJ2wDFagQ5uKJPOi1niu+IoyAtk1LjSpw4xxo6VlqMJon7Uef1XbLoyi2SUcS71e7"
            b"JJy66819Y9nwe5HN0EqAgFapu5LG23kKaVvKGcpXwlQXp8nxVSD4arOjn6nZZ+hvS+z9o/Td6kNtsAiHpD5jkORIpp4ja2KoByZw"
            b"riwGYMohyQJlwXPlHRZSepa0KBH61VKU18ouv1ixcpkw5PEevKphf12+8rpKZLmsIrrMKJLHusqA9wXJ5W7HpRCmA9XK6ZUSueyT"
            b"6GAqot+GWWCgYsnxC0Lxt6rx45hWTmUp5VSWUk6pauUWvR4Exg50MDvN+C5ltu/9B6D6V1Ubx6n1iAtCBcIe88Q94LIkvFGcB0d4"
            b"nBGAAXDluEHYIMS1kx4xTpD81Qwxi045INQoa4ll4t4BpUsmk9VIpk5x7BGHzKVe1pwbyGkETgca9G1mIAtetmmEWgKyy4uyLS8K"
            b"KxrqdHLId4UCoLWBYavc8/b25CNmQ4EJ7DILR1FK3M6GecqUmPFu8bRDXyZ5hXOJ43zEZNrSv01HkkedbZ+NxOM3lvBd9/pSZRdJ"
            b"rymEmiZANgX2GekL5Nr5ThDTvqtpBGpigCoIL38+ROSr9R+o6p2gKg0UmDEEDAeBUYzCqXWeYuQw9t440FwH7KUQoCRjKPpsE2N1"
            b"owRIoZV6r9rOWjPatDzGUjznOjwnIjx3ofuXEgrJgI6iRvoIWTWQ6OnJphYnIR+JAtEu/YLy53gVnBC1EsJD0O6B+DeJUhR1RcBn"
            b"IjyAigkh+BiksvZS0w3LdmvO9fNyOM9I26xZ2TpDe0lC54y2cAlZv730TRVlrvrPXyJwc8gcAFMxAHDKcOO10NgiFhD3YLkAQrQD"
            b"74nV3CPBJIsBI+KGa8sEkcZzgb5WyPYRa7ics4afO7fneLIG6rOd6vcLPFnPauJeFbw915WBR5K3a7jaWD4gST/SVl3hST/crDTm"
            b"hWz8CJ0f6dFmq5JR4KjKNM+iaJwpX1DWWtioTvh/Y3arIc/Jba1yxm/RyM5W8TOg0NknszbYGpc/zAkMpcdKWPmDeahO9Wov8E19"
            b"Rq9Wx+cRpHdMcs4BgQDHgxDe4bi01wLHLSMIl5Copxj1KhgUXXSwRHhCnf5AmX8CzPxyZvFr7VEjDzukGekgAgOA/hFigFuUweVe"
            b"pwzmTH+jQ0myFr1INx1uPsCan3Q5wbzLKbcrpVU3ovtPpHGkDTYT7f5dLRD8AzF637vEF+PoH7nsoRX4IObIFsNK+HGDajySfbxJ"
            b"332vi3Uc4MIU/XTW+XrWE4uHXazPd6x+MSP3/aX7mmk6Lt0fMW9/bunOpQuKoADCEu8DSOQNBKGcxVzFyCnJ4ca/I+YxJV6qGDCD"
            b"5M4Zb4OTP66cdrU/fk5iOjjm+yAP72/XSvFaKvMLVK65KYmd0WRfS1w84oY9ygLfUHx4yA57TJUuDLfJcRPUrh3khN62GBZ1CQ6U"
            b"7aG/rKaJqYiDQMvwqCJqGeR6ox/vnU9Y9nyyyEbu19iO7vjskxk67OyTXqB3VDk7s3ubuO9fhe2ZClucDCxPNFyEII8xkdyLOCkg"
            b"4Nj64MBJyxlBGsn4B4wNik+uj9MDC0Q55D9RYRtNIF+ltD6QQaAnlbKDZHpuB8D5ia5rPeSQRKFLhiA562IsT+Tau2lnrMMeD5UM"
            b"FKmUDrZ+BFrdh5lSgao4cmrLdMSUmIiLtU6jQD6ZIs6HygZxGtyQwetBm8g5G7yS9E3O9l59a8wqeAa4XRTOatWbpoZV7vX28+V1"
            b"Y+rSFFoW+/27HYn7DiGztyaB63h3S1C9mgZ+KwgXAxdCW2FUwCy6LKIZJshyUCCtiptIehRjW8dlcEhhgBj8GoOZc0Fy+Xt0yi7p"
            b"eK0ZgqXgRE/TBFujPRrlCsaN9sOe/C8Jl8fUqx9FqB04yOCUg6wQBywAuz1nXFIQdMpaVox3CrIqcTHj4UqjLzvEBR/Dj5MXCzaM"
            b"ZWwYZjUiuOfE3Ii4cH4ynpUgw0iJdCuuB7tzwRl5gj2QJ4QBj3TNjvphHaKswiU0JuXvJ/Z/OmRfrEOmsSaK2sA14yyAB4sQRhK8"
            b"sQw7omyIYbFm4DQx3Htl4o/Ve+xUDIZ9MPL/Ts/y2lxxYdLZ4+Wrko03/XuBvq2whiZl3ZX4CnINAI+yy/AcQ+MISTycBPN501eD"
            b"FwjGIGfd1PKS2UI/ufHGHJPWLYgu7rISx1CCN/7ffcIYlwLzZaZrAII4rKJufJ8uhjpldD1NfBV0sBdd+/q6/Ew8Rdnl2cnjLsZj"
            b"hkSe4zXO8Mazo+15kGn98UI25fUevL8J4rkJggehUXT6VmguwWIGRpNg48oBg2FIMxIEE/GJVUIEDRJx5BXXPi4rCP5WeJIzCfo3"
            b"6c/D4sI3ltf9nBOQCXROJ2kjRW+1H4FX7IdnHYEbL+EpIUyX5H9AYwBroW6G3RgRwpDO/EgIk+ZHVc8Ng73OF0rDKoAqXDops7LQ"
            b"kE105RkUkzq3cqPO+B7imDk9zJxm4QoBQ0sXQ6ut759jJgtY6cgb8zhj/D6+8oSaI1IYzzVYbniMdwWJ0UAgVHuClVXYa+UMwwFj"
            b"hYWKfyGUpVcbnaD7TFT8SIThFfaYO1IMQ9UDfBXGIVew8iMYBztN2W4pBXbOdzCIvOUK0YU9BSDXuBcfjkgr5Unc8vuMxnlYIAzz"
            b"BLO643GRkCV8BdqWCZs68PCmr9KXTci656C2nfubm43TaYAsvSCDu9L22SU70nRf3FV9IIzRGCff8aYE1wyF7btVmWZ9375jhcWW"
            b"12Epr9R3oWqyhkr6prAdHqVvtvRFp47TvPkSUZzPEN8coRwlmsVfSYkziEcJSI9V6vAkzChirOGGmRAfI+BCeu68jxFo3FRccRfi"
            b"4KTQELChRFPkvojSq3eu5/SDdzMHh2hIrCLbV9LGnQee8Fud8k+gin5QnDNGrJP6ssTuc78X/eAsX70B7HCrFZywIKkW2HrhwXUl"
            b"q/T77j3smg447JDGmi+PKKQa/0qqlf3xSrP5wjHLMZBd1GEP3GFEqlGss2/eHO09osNUChL4FvPOo8Y0TuuiHm/YZvmSaF4cb/d2"
            b"0dDdtHjbd2ckFLRT2b3T8PbFxIiz7IBYEPan2YEVyXyJLvFjDBSEYm6wd14E6ViMfJ3xgTnrBAMvCDGAmWOECiWFMASEB+xValU2"
            b"hlPxNaSxzzaP1Lm+N+WPyaJQVhztxjNNDxJkyaICCA9AtngBB/M1dziUEYAikrBTx47guvHTlAyFRb1gTZ6MDodlnnj4koVldZfu"
            b"AY6crK5hjK+qQYoNDy0Ev4BElsUwTu+0ubjhHU13IFuuC/67VTn2LtoyWkkz1tuzxOqZeu1IoazymW8IPr8GSXzCFvsYMfxGtlhC"
            b"rcCGaMmM09GNWaoE9xAfHCelcE7QOAJrks9D2guBhDMaAw7BKm/l17VkjLiyX+W8HvIE9svES0v6ij57rlh1maX6cp/HTLemG12N"
            b"FGZL9MW7cK9nmzgmALbVf49DHquLD8PUcrzcYrdrgZO1HHUE/eZjFlpbJqqDn0CLc9RYJMQxw7zdCSoqhiZSTYcv9s9GnM8I1Ixk"
            b"aGZ4tHMZmrbBuKXRmXYYd39tqXDoD+DcvraIb/oxHnNxf6gfg6R0qYtRDtVxKe+1p9pYb2yQMY4UWFhrgAeDFRJBaE0cYTL6HQAq"
            b"qeTiC7ltFknDF/C34hEEt+P5v9o5PMDjih6SewmPeoT99iX6KxDcs1ZhGLYKVzxm6zWUsHl4X+qFem3c4Ia7ZrhtRhBrJC07Ba9H"
            b"GGJxhBEfAvq153lrh9suvEUVtzum05R9gFTaaDtN2fGbyhdSrNvm5LtUEATSfbnroGXloGXlXmXlumWjIDbWlZEn9JQPVcQ63rI3"
            b"qoj9O4Dwwy7lL4IOjxy15d54CSRYIYNRKKUCnEDMyqBIXPN7oRVwzIVDkirqU5LVYwkmPqXe/Bg2iZOEwNM9eE8RUD5QbuRNNx89"
            b"Ax40l3uXBnIY8U7b3bJZSc6iGjqwLd8POyRDkX0fWm/PrrnLX1Y1pkuFEm93bI/xB02AohyYACPV/cPrHiMAcrQsY48nwbAzSpye"
            b"J5umUT1JWEZB0ZQaupGKyCIdWwjdvvtvdZwNrnf5jBTytarUVr9dNMUqJ04akuGqCtakdJu/V6W1AZfZqw0f/7Yn7gUCs49RTsQw"
            b"WtPgLVLIEC3iz9aGuDa0SlhgQkTn4ISTQRBvuHFGUy+I4iEwgiX/TBf0i4w9V0G7L7Qu8+tExV2APmQfI7Pw9g609rwZWRwi1K7X"
            b"bVv+Hx3bERuxNMfNfBouAIXiaeXua08utYw73z5Oh9OFOOzAC56YKkXwYLroLznfxGIcfS5uoLudYlm3azFP18PxmvvYeTHVOlsO"
            b"1NSyeb5vTYvzXeag/GjdgUTQim2y3mYnIThbeN+riBy6z3gVyW9v2JYX6beOqIg1O90oAi1c8V8GSfsksVAO5Ukd75Pa9dM6lKc3"
            b"CIe+ANRLHA8IpIQYvwcbmNZeADPR/WNmkLacSkY8coJoGUKIwbwDpHHcyzEL6EvxbS8mti+iIK6CKm40YQx446HyPccE70hHt2tS"
            b"IJfYOQ/9jVtCnp+k5AesFXV2uc8So2W2SpxKFYXReUKaZDeEOzQbq8Jm0qOcy71Iry365OSewDKk9FqtiB7kyrNxOQ+nAi+9nmxJ"
            b"ypCqFAkHmbuyA8WZSV40X8QhjVNM0ve1TRQ30+u354izfr5yVRUrcfOOFbgGrdEatIMj0wqOTGtw8wARVyHotmx7s9nNIJzWm78L"
            b"4PGg/eProR+DmYJiMJbgIJG3DMflATUkWIM1plYhJnT8x6G4kAheIEklckxSzDHiTsS55Y9O9I9O9I9O9GU60bTGuuPtx1iSM93h"
            b"B4rEAzrRutFkj/Urv19PAVsjyh+d6GdyO1TSYCjyBiMbY3amGI2rbCm9llgp56w3CCg33AgcNOE8UA/SysQpKgX6Ilj0h0RCXwFN"
            b"0wfq6XR0wgkautPevLj0eAya7hr6LoKmx2p8qEUpk3mtYVstDCPee5DsnJHP2GdMiKxh02oiclo6WZJxLldOx9848mRYVP+e7EmJ"
            b"kQvQe+p8OevdkB813EcN9VHDfFQnzNt3/+W5dzPmX6LS9z1aUdoIuoibPoJIf7AV5RQiTbUy2gSnFBMxXApEa40wt8Im8XjlFFAT"
            b"mBRCWEsxColATglBo0e21vtP5NO3UFed1jrJ+/qdnyZrflTorIhL77CWnvNwrgAYOEj8nQWm4og0zCDuNVNfkguD416pAe/4kTqb"
            b"TyfnfkR0SsawwQUEyCit0xrbeYY7FusMIWQctv34xvnZ2CYLUXienylt3nS8BSfSkhnRuuuvQgDWb2SVpOi2+QJmWVzu/kbuHrXZ"
            b"rIiNKihKhSFcbL99FfMDIkwfa/+j1iaVpcCoDJ5LZ4RN0qdca+8Cjh+qpD2tlMZOS/COUI61wixxFnli/+/4im7kq99GbXSS+J4I"
            b"pr6dWk8UJnkhyYDkjQzp/W5R4j1I1Y91TYtpniyvUhcVuzzJ3mC6K9xDkMnu6NqPdNxrSPNXdrnKjycrfrynKIvyE3xbKYXXuQ1e"
            b"+3ZekzPzDo14LJLKqXBfgzocTwDV5hGL+Mde9B3Yi2jQSjCtpeZcgWAGI86sU4GE+OyC5Z5aFuK6UQKA9Qo7HAx1IYSEg6E/qdB5"
            b"x9M/9uEXcxcXS4uDrMq04edqJfZ2efVexgL2hAViTQ1xaH6Z766MNl8a29ZC28Cb7vEjaJItiRSiGhCMqAqbfVE3U56m80THjmnX"
            b"BzpNpYg8vLQDI7Cot7ZnO04ecVDF+Nmy5s254Kw5SFZK2rKJ/4/q2fsKolbXelZJ+2wWeF3a5a+I+cwswEB4RrhlLj6mgFhKfmNB"
            b"tA/SykAp1k5JRpEmPvA4UwB4JJwgHogBJ74dndPpcmL70mdTx4WGo+Gs0VcOKyfSxKxHfii8hB9td86hwFhFqARut53OCKK65Hmp"
            b"DpZpQAKuT86G1cG2okpGNdUKBLlhDGeXXc4qc/OTwmuZtnAZyC3jc6ywLlphWHViL/LA6JqN0mBInB8PJ1hG1O2SLMuA2pbPtcD8"
            b"OfrqlaOpzsPw/2pI+tokX29v+RqC64QNwZX/JngL+Ks3dYfR7s/LWY++G2cszLUeoy/y3quXriqXaPkNvuCjxcKZMPDUH83nMExc"
            b"kBQ5YRFCxmJsOeOKy+iZNTPEU22VETKF8dLZ+JfAINAAKgiF0POi2iPfe0+T67wZ6Hlq6V31+sgb90YRrmPSBQ6aL+OMy1Gw+17y"
            b"5JicYdsldzLZRwx9I+Z1tUXooDiQCVZyIp1RiqoB4/kZimn29IwjdKbunS6tGD3b6RMfmScgH4O8yAU495LnrrcLeFxWMbRsACF0"
            b"A/MdIOAdwu8qgO9Hqlz9Iy2r1XcSjJ3hQkrsOSOaBO0d4577+GMEJAUX3snoUHlgTNnUQ88cIlIJrADTl+qP7Hu0xo8Flg4OcCwa"
            b"db2VZ5BrOG9uH+WHa+AarBmeuwqLh4pmo7vHB0yti1suyoK7xOzSoHNEdCfL6OpQo+06HjAshum1KoXOdxihuWHJBs1vy2FeZEO2"
            b"gzyIDb/NxFqwFHups8v5F7Ok1rKjtqsCZ5l+5Tl0+ytkuc7g23AK34YT+DZM4dsNnGTayrnrcx2lutZNTuvN79OfP4P91bntWbPP"
            b"Uqa50OzzsG//a7IfRFtpveDKB+4tgzgXSIrBcQQu/kUaynGIk4OIMbWIMwrzMXThTFusHSU/KQd+MR19tSfoekb9ToL7jRSvZ12j"
            b"8skE9Ui9S8w7RsupCw3KlUR5GsE6lypM29FsN66fN7JpzmUjxFGTjt5AjN3AimW+ius57HgFWw4bwc7ROkNBFjNCX2jleYuu7XPq"
            b"tXOVXBiqs9xTbPnLdv+jbDfnPoCgQWCjOI2e3DDKQ6CKa0cAhDFgvOGSKcGRtWCCDsyH+FOJxu7/jrm1IsBiI9jJT2NGZRsz6tb3"
            b"3nQv0jGPWPoWB7iYfJgkVibWZLJotFp6yoN3keDm8yWTJylbAcv7Mi1HzlY25Wx93GdzZCc8dMr0jTJ/nK03OVuZj8+I4dQxo5Vl"
            b"GmMhkUJCOU29MjYuw4PU8Q+Ipno2YE3jz45Hr6c1JfqL+l7eotEyb4dpvVl6aKqnp76K5ae1igNisceMpeuyLX1luiAqt5/9RoXX"
            b"QOHkIiCIFN8sVxhp4+1EsSExzmqCsQtsLYfWlniIQoVavFLVGnJgSxm26vDHPT8PpLH2sx3zLmlUuPJdd5VR7gWFUIVx1wM8XO2D"
            b"L4WLcAgF+6DwJwqdfKGcye6zlLdCI2dRiP8w5oESLpAwXlNmqGaYIaklktg645kRziuKDQrGvYxhngOY39pKUqKxkzhsxl1XKuzN"
            b"Wae/5P0UG5NbNJBsPRKrfvHQmV5qXJFs40MSPR/E2jXxKP/QJR/yoUgqPylYn1W1HrK7Mdkm3TUSt3rjJjGcDZIdbtgnFojXTe+a"
            b"DpKTHl/Dlp8Tklt8tW/PeKGLlayq+7LixIcK+gV1x90wX1ltPmzX+B4g3VG3xscBtwMfxoyPPyGtg8OGQnAmPoGp11hRhWLg5YT0"
            b"hrC4sEQa8UQ4i5jjhhiJqOW/z4fhLaji5Vtiw5CqGDwMvdJCqgRVBdC05avoIUTLJnXwNVRAngVf56qiQyebD5FirwsO7NCqJyY+"
            b"LrUpfwffVXK+97zXWNEjl9KrAnrrvcbgJJiy158VWXo6zT8PdsmDhfjL8dY5qShiXgumkLHWaK60ZFIKr8EYo63nnlvjYvRlKZUQ"
            b"kBfa29+EPLoMGboF81lQNrm1lfAGLjPsWrqD4CnYnaxHuN2caTtUOX+2fIjcuVCDv4LTugauWiBC8ZWNitVtfy3a+mvFJh5Xt+m1"
            b"hasCsXoSfQT3itPzIvJYRn5bpVbvcLUWxdWatds+kB1UXrH7K622/mBHb4QdcYKMQQYUs4CZZoIQpYw0IgTJgHgqjLAOIU8l9doG"
            b"7ByzRHFQCAj9bWh68aRc0Rma/jpMHt2Gyc/Q+cdmqXeh6dEFNP0zCPleMq9IfdA2+j40rBUofcJvkK2itRl2JYhksqLicYmba8Fr"
            b"caR3J0snwHOySc+xz0zA8zAFz8MpeP5RfCoXj7Krz1Xg+Qb1s4Hn4afwN35AQ+ljyHmedDiCF4qYGB1ohygNSsQI1XODrQyYOTA+"
            b"rrVtAKuoFiZg4x3DPDpv9ESZV7xLP6lelcP68xw65bfCR9s670DmaBNUmskp1YJFj+SKjiH1mLCRrOxfGNaVPlr1f9glodGVlPyQ"
            b"GOhqrHAw4sskJHh1Y8V6OU0+IjEDRDO5zghiDXdGA4xXkAzjXLjOH7uqURMAZ4t1pf4MkvJ2y+i4yiun6/i1SZRWVd5ag45WVV7a"
            b"tC09FjH6AXJFLQVMgTfiG7JEb6z0cooxs1ZywySSLGEXAZgRBDtmlDbcSkuFxQG0kF6jGIwqnAgAiLXa0k+wbd1btd/UDzqDp/Pn"
            b"uGCuBF+XJYLmNLLoMo3sVYz9kDUmXUg68IZz2QgF91M0PTnplzZrOiIvEsLkXETx4yXBsGQjcA0WOtIAJOOd2KW6NTeIXY573SJ2"
            b"mdysA7HLTT6vm42gcKJWcU0yviGC4fu+j2NkqAWIThWLqmxr8/djbvaXZRyOzLcN7vHrcxGjgJhpiaV3QWsloichxNIgkCMOB6wE"
            b"8ZQmBDxohJhXTIKIKzuqOSYQnITfnZd4CnT+Qv7gumzztUTDZfj6aS++OKL2V507vlCjVBPZA4XpOdY+WyQppUZQTq4FRXy4unRd"
            b"ZRBHQTwYDWUmQLde7jRB0enP3UWrE8YoJff6+7Os2+Z+23ez/tNitycl2ndbf/82W9Tv7nX4lxG2Jn9pijemKbjRIWiLnMYQlMDW"
            b"xTcgjCc69f1jzpUR2grGNHEIIUs48YAkpYmQ5UeU2QaEtAOrN/but1V6WFVmWOczry0LBvzjpQA1L76NiMWhStB2YeuY1XegQDQh"
            b"NoeX2Aa4KgOliLJuoTAcbbZbyL9qfYcz5aSRyMNdQeY7cPU57Tg7oR1np7Tj7IR2vGnrH2s8VAa7NNxj2vG/Atz9ApxmQlhNA1gm"
            b"tRdag/NKSs4Zlzy1cSJpwCGV9D6p1jy6WemkNFIQY82/4h0njyR2HujrtAWfZ5zpyPsdGUbI3unwjALPQW5zJ/R7EM/itnngXGBz"
            b"OMpRQFtLrB3UgmYBbX+TZgEtrrjD5cWAdijZeRrQ8vUsvTzQWFD5rrrOkqu453/nUe0Zq8pZVMtOo1q2RbWNtOYe1TZ6aVtUy65E"
            b"td9EdOda4uG+5M6HIltBpGIiKK04JRIzpWTARBpmnCYmblLsjSTEQnTIFAFRxEEMgQMCDvACgGzhcB464K0WcFvjDE6raTdd77VA"
            b"LhnQC/RSMwYQeeZ3l8hOPlJkpgcF5PQ5pnWha+2cfDi79FbpMAtzH+Az3FnqsEwmhODNjq/J6Y6FJN2wbPdK5QzuYfTv47VgmU92"
            b"u4u4sDOM/hJvfnsikcqVLXBs/CWkIIdYUSgIQmEJSf+LCB7jvwCKOiSoDibwuOKmxgoZPRjSUnCuPY1zYXRZPD679FfEitcCwcEi"
            b"lRwEGy/4vrk6TR9SlqCSAkJHFXdCntGxua5Mk01KByUTu26jrJrW+z3OlGeG0Wjyxylilpjx6quT+wS+3ol0C7LVTAWn/0aTKE2R"
            b"tMEMLzusg9ouoOe3Slda7F8KFwl+jxjjzKmy1WqLAvdPZk61biev5NY32cV9a2fv+yFSjD8tKlTBuOg7XYz7CEvN6DqVmDyND1WM"
            b"E+Oj7hnGBnx0sTQQIkR0ylQY7qKhJX8O98/h/jnc1uGm7/+2RMzB4cqpw5VThyunDndMjTQiRFodrvxzuJ9xuB5JEhQScZ2NuNFS"
            b"Y8k1iytuoyhwjlkMexHSOkhEkEMSA0rkqIYHJ43/ozvq6I4e8AHVKNhswivxxcdSLmMMQA+CZcuyW6zUSeKwkH8bz1A6ZW5UY0Br"
            b"dLCoZd+7fZJpaleL83t3J8XIL7NiSVjxgUvimA2OGw2St3uyefWXomH/OI9aJKzwYIUwiDgNEiVJb6SEEi44K2JUmXojwTDjsaUm"
            b"WKowYjr+jRpDnDLm1OPB2OctzTTozO09FZOeeb53SdR2NEaHHF36jC3pvqWV/ZDtSwYM9hhvYkdLaq9t1RrrDeKCoL2AjGV5Xo9f"
            b"gUR9IYUMOI2Lf1a1v90p6K60KPR2au2uqnptNybr7gakMWY7iig9IxqINzOb1DnMG/z4t5KXYxokmJIdnX0yo1uCLQyEzc39xEbU"
            b"EuSNfNzjeve5L0NdtFd7Q9giv8XHSTA6tZEyQwQGhpXAJOUrJZUGM6skaGtw/EQb6zUSXCup44IjBoJpn59EWvx+NuKLNMhX+gFw"
            b"XsKVmjJb8TOd+FAXcy08KGsBpXpa256iW6KA+eRZ/I4JVfWEyo3ApW8iTXZ5FGk6VJuTVetKlR5UkK625PLFbWIA3MFxF9xI2/+w"
            b"tjXwGrHVUB334gF84ZujSqK68bdiOu53SYZ5QAwU8PpyyeFCyx0phs8SF6fv7h3MxbR6RztOY1z54PaT0yX+NOis0KB/zMXfkLlY"
            b"QnCMGsZiVKsDJkmLwqj4o9SEeScSnwAVXjLNDSHgLdLcea0VOCOUFe/Fir7IHYWn3FH8QIE0jBqfUH4qnw7CRH7G6rl7xCthZMv8"
            b"mT4/lMof08sM08FXS+U5xCRllYzPmU4Xm/hCa0Po2oLTx18Rh55Gju/95MHWt49DpxxRn0RdHqrnkgoTfIw4NcJBEKuItPHVUiGD"
            b"R9oEIMZZHF0U5xCX2sgz55XFGjMskf4ilrtLqHY4Q7UP8epXwegzjrczyPpGLNdB1l9Qpj40u44hSN3RkkE8nYDdTGwwy8ZQbGIb"
            b"BDHVUIcMiYePNSSoVSR6aq0Mp487yYVoga6SH2MtujwGyGEixmIdzUJKsAX8h71wztvLTCMFbP8yxOjLKDaFc6rKXH6KCHROfTeL"
            b"99aCDK3saEUXSqu4knbtoiecUtXmimSHh8Wb70Gt9wSM/ZOke6cwdsmd9IwZrp02yFGnuVDE66ApsyQgSi1RyhOnXHSpThgTGHVx"
            b"2yGpDPqEfN1bUJQPaEnuwtivI88vIRJpBRHHxzL4AZc9g0w+QL1PhDev4UKvKr4tapx1gZytbfDjYeaLztRSRGzUUk3hGx92SsMt"
            b"9kCAws4vtVfjh/sV61w5j+Hqtt9WyB8oF2a7/O21BfPb2M57JKbzbqIzgtN3kEjtcelRLe4uhdQ3hn4+LJ9/ESh0tOaWxkXHga1l"
            b"yHJEtCFcI4Q95fFplEAY8VYJY3lQnFCi4nsclASOmRL2QfmczArocl5Cn9TPvyTmrYvfsjrtDtKOjxYuHRbLb3+sMLe4TYEW25NY"
            b"rp8DxtSr8UjJQBFZnhi5OjpYwUD1SJNZWsLHL3IzhxOGkEsr9DEqanZEtBjuLFArD6roV9+ZAwr2J6BrYoX/nlYBivf1ngqQrCri"
            b"slEBwoeAdJdXpr1g/aYPRKuwlTayy63izwbc/GE6QE9zOddVcLS4LNxVjkaODm+VI6hCSRmks0m9DKX+Gxs9ldeOaZXIxalFLK7N"
            b"gUrvvZDWKSqjyzOYGZCJrP7ncUO9wlJ8+P0NiYrJRSH4Y8Pi5R72i+v46zRY/5CeeqSLKXc+qyUjSyq0FB9zWhfjG5RQZfRlh6vk"
            b"TqIid8JsRCbdYQqyWdJJfo4M6n5/e+F/4rQig+K0ElfjNdKI7y53MdyLS5xWuvVzFeWmebLbXFzywg/Fe5Py9xP7PyqoL6eCkl4I"
            b"gy2RYHHC30uspdNWCwtgkA0CESm0ICQu6wjFSZHEeRSsD9IgZZ9BSC2Ew+9niBJzhih4xBD1ooLyUL/oMrPTuzWRj8iCIazgTnUe"
            b"FoQAohW0a3zYUd4ErZX8AaQ1HTLhFZBgvdjdhj9oJVrEQvAvKTmKzI33SZJ06TIxLvU81UALKnG6br98DXknoGxlfWk17dod0piK"
            b"6bPUUfdA/LkiWomn9LX/EQBVNkiAdnt2tHOUwJBH6odjA17iiHoe4EXyNpmAvXzwLGAuHY6PAxKeCMed4MG46KA9Qh57y4EJbJyP"
            b"JoAMhyCCdmBY+EgmuOXgezMT9VPh+zgF+hXh+1DjchK+H1KaJ+H7kTllHL6rtg9iD2LjbTnhc6WjGzaK4lVNYtLjzk6i+DH79aMo"
            b"fjsbu7YYmdnfjvrHlK53c8tPUADOg3l5EszLLZjnTTTPt4mgvGvieV4fmXYdBt1mFdDf+/u3T0E3sfrm47eAntTJaVIH9LROTtMv"
            b"SU5fjukViT8qr6xzzFPOHVLAlfACtFcMggNQSrE4TSgZZwvEPKdSc2QhRFfrzI8gEnwacvEs4eA1IkFcGlFT6rlLJ42uYemyRUq0"
            b"pFKkRTOvrapItm1we216vdhkQpJpBTTbcyG98QojA9hwZGvj1xII4U7YORvG17V8mVAdy71Ys2/NPvnS8oQmeY+ILj9D3DW3lYMD"
            b"MJCrfdM82x4+HbXYPsknWKbbOxytUJf/qu1ZCntB9PK6e4x3rWBH7IXcsRdNB9h4c8VeyD8WwU/ALxRD8ZQJreYBUy4Ecs6BlQ45"
            b"RY3hhIXoOwXXOEYNxhIdjFGEosBBGIPfS191L9w+j7VfplwdxrungaN8KjZuvTirxWRuSwlO4nt6YE7N4yewRow7VIOMlGa2gByj"
            b"chXiP1ntwB5NErOLyhRWaRBxUmc9omI+oGRcBgMc00HmenT/smX2wi9hKuIjkhRwbqMqeAOr4I3D5A1NK6+IsPgW325ErdXbwtS6"
            b"SyDwkSDCCaytMtgRbiW+pb8tvP13QezqaAWSDLxg3kjnk3B24gu0inGuaXLDCqTixAWDpWXYYeO5pNxYRrF04RmExfZ13RProoM6"
            b"5qHt9pwH5rEO1ikVzEoEg0dcMBOg8eP8ilxrdOopXF3LknrKkZohElKt/C6z2mMbdpM17D7WBPnO6LrztNCKnOAAcMscLVkcl6qt"
            b"pWTSAJxNRAER8z0LPJaeSZdV7CpW7Luc2J+lfimoX1whgHEV09YC3PhC/xmtKBFoQ4/Q0Rj0geuG4yibrebA96F/aWLVWvTrMwQw"
            b"7wF85PEt6YHVvyrqJNE0MR7EB1RbHv+jOE2aAiipHgZFicNYK8kkOMwZgkAEYtwZaty3031Z3N95VW9a0ntIwXJN1GVFtF4UdTkK"
            b"hQ9FXbaR71yrq3fjcjeutbx5D6sgvAw0EUvtTWyVKO7Y3V8Wb9khgEtgSUYS21caVZYhxleCq8wCqhghxkNFSzgvG1Xwfb9+1kmW"
            b"RDI4OUlLywXLoPZiXhnCVbecMsj8VoIBH5IKe46XtuW4zWqsKy6Xop/ckMW7aEyVXqg2q0RC2+zbWH62pLf5dkz333D50Yr9d/u6"
            b"6EuV4Vq/1WFBD30UVKyURnF5F59iwx1zlApvAhR5cCykTQxdOFpIgTVYi4lGImgcKE70Dfb/IeS9HH5eCmYvxqjX4ux7kewRFjHr"
            b"Jbkcv5eD5Rbk/8uol+BbbcW0YjWkDcPhOByGadB79sk4k9AT3Aw+/jwE7v8y6DXUeeRT2xxl8R/uuQSdcrPU68CNtSrEyFgxjyRj"
            b"SOLAhEJIS0OwNd8K4TYLhN+nflievO3S9utYfuULWmrDaYkVqdXQYGX5VZigueZ0DqhqsOuitwEJWBXIXiIBG5JFdOwNRZprmNa+"
            b"TNF4IAGrI+5umpmSgPWGPQnYTUzZ72MB+ybosTnR4ScxYgdcmHKIemUD82ARSqJWVBNpuY/BIlUaY8MgQXcRaMQcV0EApcEqHYhy"
            b"jv+Ikv87VQGvdVRMpKe6UlH6/CGXzGlPxKi9IctkVzSCfBJ3zT3IuAlij0JXIu55CLx4HMrX466r60l/RTpkxqYRxBrxq2lbCVuR"
            b"XGK3nzeWFMM8oCfr/T+VheZHMtP8qBp/EMaDMkZhDtE/EhQYDiowE6hTCoFxljGnvYl/CEECoUQ67OPPj3hO/o/7IC53OVxIBD7u"
            b"qfhEb0O2SReJF9zUIQz8sl6IwrqQXSPqwuKx0HY2zOO+2J+QrfKanhEkDyFn32qReBmyYSPwqob+/LCivxmk8jveedapANO+B5j2"
            b"PZwdbe1nqKPU/W87P8P+t7++h/f3PaiglfGKUIOYkTou5oUAzxQIwhHiMjGPMU0EFlQQCtZ4Qzm3KGAX3ZP5RJp0liP9EvTrrkxA"
            b"FhL+FuRUhEAXtNASs8qWoPEGpQIvgbAk+FHoGk+arGQRPyULmmTGsrUWkCqgLB+SSOwfX2CRWI2nNBJpdJlsBmHScs08RceYD5MN"
            b"nmNGzCe4U7ofMy/IKfPCs7BVGED7j3/9PmHofeqFR+HmW6kXdIwUkfHSCXA8IOWkldRLoQNFHKMQiEQGG2+tBeWs1Vho0IoGMBg4"
            b"++n8MX0lYk00VlSCZ9yr2wNy/PGTLEuwuDtaHr0T82u/arYWigCT88rHJReaD1NErVb+FlVj7U8gT4M2g3yQdDBV+BXzomAwT7R1"
            b"pKHUVz5E7sSqeWPWitSHNFJnCcYPpB6fXkR/EW/MS3SE8VsqEhgD9pj3+y9lJZciEEMlsTaAlkpY5phF2gkUP2ZOMmUSSjM4rIRg"
            b"kgUqrZIeE/Mbe0//YQvoVX7Ag3LgBcDoRg04vC/3uk/vtpESXrWRys257fj44z4L7SBK3nIp67ddRuORFeOsHRh/O+teau81GjeU"
            b"0rUxgKJeo+DkurJ5uSSGKZZ1FYsuVzdpYM2jKzu91I96F4pEq+CSdoGmrHgPZQNSmsOXZp/0jAEz3Og4IN34ZeSP4T38lU2n2gDx"
            b"gthAQGkrvZecEAlSGUsIio9vjH+1115KhnRQRDKMEBEehNTIqv8PAvAjenRKW32367M0cuIVSbrTZ5NRM+cCDL3KoZ2Pm0rlQCjd"
            b"0KetfXuCa6jPfLj0KmtebnJ+5ELmLdAq2vNo8Je6bPPHZSTAKGUrnlQOR1QfPlvndALjwOqLLZNCwyGXTcQrROHxIbtHD1b1mzZv"
            b"VuowqMgGdgKwMaKUTanHz5gae2Ht0qRaS3D/kYW/r5KlrXMWDEnVVYYkWBTif5GnSoF2mkWXTInVqVgbo3UvLCCHPUbYKau5/T+S"
            b"1b7e13RV5JrTqgkJV4QmY6nobJQnBEwX+MCJYOBMhAzuKoX36l07u81RkUtOdRehg1yVa4hB9a7DtcQwLRFX+nwRVlhUafCWttkV"
            b"yqBXVsvGTRFKziqE7wCW3kpozPS3zj+5n9DYK1BzxFSFJf2WsNLfIaWtA7EWRW8q4pJOUIWkAGqN4g44Z8ERyyxDOChPEj244ZZI"
            b"LzU4oQSj+JdxgV/IWOZ05Z6tYBOzUcLyUDTauucXrNDQLOUESHLSoiF82nwxHQXnBTKAii+u5WF6L8aLFU4zyIhopXFfySjPSAQw"
            b"ekyfm4acLQkpejVl0OMVxbXkczpQTio9xw3+jILhWCx7VqDCVSiJmxBz5g0PzaP1ZpVX+CMGv5bdNcg5LSmWFFHjeVLqcswjRXlK"
            b"5RoC3nDjvI6zarDAnGWaeQUqrpssxeQfNyJ9lEP8RLbraqM8v9Rr+qgHaVj65ks4J/ixF6uXWdkzt7REdSuabMyzXczSkVPYhpYd"
            b"cNWH3+m90sVwFAW22etikV4rHMPYlpe4d1WhqbXBustLB8NZfQYLsVb21TT9m4xy+vegOdNX+CdqMzdwp+nJ+HR7/ng1f9aev9O2"
            b"Dtvzu7r/l7bnf6ruXwWVI4TT1yACBm1KhrGgGDLgFGcScyF4EuwWNjjNVUCGMMY4NyiuGZ30hCT5hiB4XLTHFbz4bQinrjMIn3bt"
            b"7Jqrx/56VRqHBKK9abcWzyZiF4Qeu6LLoIQUSLKUN6AD+dhnshWZ9rSovoqG93QwUJHFiwXfhF/HIItO9PVP8vW7Aeu/OaLJcKyj"
            b"UxJWUdDCGw+IUiEoJS5gy0UwEmPmHAtWxpCSKOdxDCipSH2W7hM1n5ql+UgWAo/4QugZX4h4SNo/sGoQ7fI/dRjHdVb/Y42p+N1e"
            b"InvS29RzW9ASD1Z1lC2q28Ca6lyWtrllt8RkB5KvsorsuksqBf5VUVZ0RKKbRiDuZF83AVegrAEF1FoB6/q+03OFtajTqr9u16xO"
            b"1V9vkoQQfHt5fcRInX0yX0Q/WF4/sfXd2UFa4HuOPAY5yOJ5v4Ad5LSWkyiZaIz4oqdFyNEQiFISjNNBWUykQ5xgZizhoEnyrchq"
            b"Glc1RmmpHce/m7DpMsfSRc6mvia+6bAMrZ/y2CNHfJluKVtkQ9Y16I9M2YI2pWQgwN0P4RoJ1aNmpv62ZymUnHUEaOv7ezfr4TvN"
            b"Dx4utR5EBiPqPG+0kehpz3uPNQ9XOUvc5S/HutuPyZlwRc6Ea9a8M5bnETnTYvlHzvS+Ko8JxBsdLI4rbJR4RIRzxiiJQFquokO2"
            b"MfxVRDCKvAiBmxgCa2aRkRZxRP/PqEP65k1yDpMfOEy59Nvz1UDNuNxotpKAVpxOTePWn1lmpZNNEbYy31IPzxB90M1x88edndkq"
            b"YWF32OjmCdVhCGmkBTGKhGx7TGm3OhfJvwq1f8utApnIv+wnWUMwjy6cUXYrdZlzJFv9p3lXxAl2buf2nSymsHI3t+9KE1OBDm0B"
            b"av2ugIpy3Tcfqcz6lR4Jryny+R+3yFVuEct5AImYwtggCZgYwVMbptaaWEotxRYpzzC3gnNmEOFKMJCGCwzMhi8NQt/fyz4K2eiC"
            b"1E+v8vjbJ70qa+FNJIqPWrp7a1XcJMQJpkGJsx37UEHrc68AwWtr5nbhYlea3A6dDpi8MAjgtKdG6gZSDllISBQ/dsWXCbrdI9kV"
            b"v8nQ0lpe7UKgs89GpS7OEan5lyrtJ9LLamfT+EorFlTWgmPJRb6TA51pn6oZE9cVmpN8D7GibIlR15328eB2JMWW5J2k6Ae/AZ+6"
            b"XaJludgN+n/XhzN+s6t+VESCaRx71lU/J9zbufRzHL39jZ9tbjWoY1f+z+22HyUdKu8/Eo597P2/AO9vhSDaG0oF48ANCVwwqbly"
            b"3MiUtyAusPhk+jgNBAtOxvkBSAgOYc2IMb8CZvpUh9hLSNQulcErX4zmC/qt5epQZBvyWx2lDmW18uc1bmAJuM91AM4opkrKBVeA"
            b"1elF5cMltitgpGGXwnXbVUfMVUYMBHF5oOc63t105GIbx1QIFDa/zncJm+5eJ8t0DS8BTikomkp8d1RTis7JGj3X7xYPu1E/79tQ"
            b"xFOyhGJ6374rXprgNa5u3034TtaAunPXMo+wNfnDob43Q2GVlsxEl+DBABeC2mAEUCIx8tZJJRxnOkbnjonoq50GB0x7z6O/YSGE"
            b"T3jhFzmhL1LzvUIGeGCEfmsr7aC0VVea+rxA1TGAl/lhzXuotTTFulpYPjSRTKyUB9t4z8+V9ijnAQ5CDVpe9/obG+Q7jqse2Dms"
            b"CX246qkZW+n5QgbXBNat9WhpUiNASqxED6uqYk3VRpStqosesF/nW1TM277bm579Fq3VLJEMU1jtORP1LGg/Mk8f2wrmW3v4Xra+"
            b"vVtvQutZy+3i8C+03H7U4V8PwWOIrTwRXDORCBcwQkFF3xAUltRojJxFDklPtYyhN/FAbPyrttz5uBf7hs7/ig7rtfnhqvO/7tXf"
            b"Tc1w2/MPpGynugB0I1GATRYA5noD9+asctBEOaswRrUi7lR3oFgWcoeL/rcY5hExTNe2432vidh5vO3FnBEMZBdMnMGAi1n5Dl5y"
            b"80sAf685l+xECM2brTmX16kYvny25M33PHnzqfxPNpnx+t2sTXfTgGk2Jy29e7zPflBk/zunAK8wMdJhbLzDRPn4c9FIxt8niWtR"
            b"pqlCcXkqbFBKBGSVlNypaOYZAi3l/zFt7QWq2Rng7jGZ+EQZoG3gXdp30Uln7j4FdxIHh87cmRhCOlvOOcPuP/dkc2OpYJkHtoOK"
            b"aho47pAsV0Cy2vapQM6DHM6gQXpTMdsLFw96pHs9hcM3MmlHbkoXzZWkC1g6Up7krsW3iqRzuMgjkcXRJ7haDeCJ/OJOsjMGl+yQ"
            b"kkcR/R997X362uifUWILx8E5aSTT3MdYQWIZ3QulUpMgIQBWSINh3ivKlKQ4/lCssxzM/7ObfgEY/ayfvjA1LCQIOzJDHFRwRsn1"
            b"bkk1BZIvOfDj7bjvP8UW2asBWfmhXSa3CBaXSzf6tWoaGjvckmmnslIKa1nFD9jAoVrYp4jFn1QkrxqXeZNukVXQXjOhFXXxHfrS"
            b"EvVWQXrzZpxVr7Ln1WZdDf0xOJaf5aoJI5IFkAwRYZAwShHktA6WW+WV8kJZBSgQLzEIwcAAcBlXxElSx+H/Dx6zk47ouxxgh6aS"
            b"AStEQ/mwaM3OTnqN7us2vVoyXRUnOq6H1jgaLVxsBPBqX/Umtvap0pgN4+tart1Vabcu8yNVWc5DSc5GvTCtdTJKt44hRNoTiMkJ"
            b"smUZ0FfRlM0IytmUoJyVhDivM+I1tRmucCy42qtKb2zh72xzTXOvaQ76R1T2xuYWJziPcbDwKYkhCaJOAHEAwrDohGUciHQOYaLA"
            b"+fhqwEcPyzyh1spg0C/EFb4x2B3nq4fh7oyUckhxcQNPN8pBn2nfHJpQRN2H0vEEoZVVmOG+RwRGVBYV0mRDdJxfaRl2vnucXtHi"
            b"KYblhiuCjwDLgdqPVIsxxwveZIwk7Hct5ulyOF5j4vUWwITNo1jmO/YsmDA/T/cy3ONY+Ty/vaAIeQ0pbD7jNdyQNxSUdLA1SWdX"
            b"BJS1j9/JKv9QhV+NKnRGCykp8YIS5pjxiFAegrQYkgYbdQggOEmDxjgYJY2CgLw2yBofwH1mTpjUOu/F4PxRGH5txlCHpfFVN39J"
            b"fq3o8dRBdn3lD/jfZ5SYG597ySFtGYs10u9nmEXbCFamuO7LWTWOSnqF7PRBswRLMooHEZI05Gub7fESy1jTN0TVwnhcZ50nuh3Z"
            b"NM/NDIjC470aarg0pGLL4mKSzZjqmn3SdF0U8MqsKlr8OHTqcIshEMLRgDWkZfFMYyiW2wxxN/DnGOg9FCPNsO7K1/MaWc5rHiNO"
            b"K+qjWe0UTuYW2OP/ZrP0Cw2Aiw3k5QCIWYiMf9eq4ME88fXrhdE8YVWcDYiKj6xiYH3CnANQozRCnGpliVVEW6GDj1GZMlZ65AJn"
            b"3CgTP37AljTkSnoTU13VJj8FqdcohTdx2B35lMjRsW+ehfCGeakXhSR8XYJI2hoOgvqeMWnAeHnkShpwGp1zPvHjSmXO/LQ80Khj"
            b"VxLZWaJ9sOPlUrrobLaTK9Gt1DIhVnonrdJNtpAxE9L6Ca4+wQ95RM4YRvAj6Y6fwKu0QD5GPZePmZVQl2de+ynPM9U10nvPQjss"
            b"PI+RrjXB+aAc01jj6N2cUAhrYSiTSiZwt4vRL+UMCSQsQ4QR6qz+bWxw7xOKvMSXe61p57oq5opXbtDKE1eYjdSacVimkJmU+1qE"
            b"LBjDscJnXwE8k2SPtzCdOnq3ZQKo5d6h56NLJbxk+awbu5NImKlesimp8JlWxQx/wXYw9EHFov3rH0fcRY445xiz1BpHkHGOEI+l"
            b"8B6kt9gbw5DTxjqKmLGWIM1QYNooKoz21ELwX8QR1+d3z5O7X4ZPGCWJ57xz8jFb24GV/RJb29XO6HNStx4CMMA/L0BdOcJcLCpv"
            b"qXmFoBrOtnMaDYAa2XSlgatgajUFHBkSh2bz0glIRcOVvO1y0GxKlrlU+SxQ4afKaL5bWvP7ZU//UY509aI++kYtmCQIe/DIEwoQ"
            b"HSuKi14rQggSWQ+KAMEpd2qJCQhiyGiMC8iaX1gru5P5vOi+Jvo6PZx29VqM9tV3eeQk5+uSegUP1C6xS9alA2avyFY+H9p6w56y"
            b"ONnlI0efA0MIboKsw5NYs6tgZL7MbhgAd9RRy2+qBS6v6WDOZS+gyUdL82JZekckqiHPi9KqGLSbJMM8IAZq53Re4iDUaxKlu1cM"
            b"n62MyXvB7Ng/synIl02VhB4Gs129a12U9zUwOth6l0TbXy3sqRxnSD3b8bHTxEvhwBFuueCI8pB0OSAApAZv5gKh3GpmjbEo6y3H"
            b"x1n+Sr3lNQ+wlKfQYaCzEtKwLjQuIQ0JMP6h0PMtteVilNMNhDQs8WNx4bvqzFhW6sxoHXWRq2Nj3ePst5MxwxR4L3I30TzOoy87"
            b"cCoYbq5kmLxOX2MxjS4csxVQPC9jFjNAr4kq34cW05rzrnlTOrk5raQ6eCuzzGvV5OqzxZCvPSD1uzNipaX6VfjxljeVScWbN7b/"
            b"k1r+F1LLHnFlc8ELaU48IUp5qrx2HDjVEKcAxMEQYYFxLGNUYzEJMiBuWADE/kASJy167xFyvo6ZOxAiXUJgoIJmazWQ+VEG+TYC"
            b"5D5m76xv/Nj0IQsHag3aU1WX+UF/Kt2wLJc85f0/SDUlw8zcTzAWNe10Pd/g4feU98iji4/Mmixarp8Pdbyy2dIP/yRiIj+/1wt6"
            b"czGnuQAUriF0uAXXrTyqDanqdkTaNAUONKEebtaM1X84ia/GSXgsrCCGWKNoECC8C3FVgUX8n8PxaXcKaxVXGQS0tYgqoxFyEluK"
            b"pdPO/8Y1xMUQ/R7bUZ/lqVReevX6SysYgqvkOCpjXFFb+Bgu5xxQtjzivPLvoW1+QcvCZ7lH60hgvJYqh0yvsHcQzg6drdKRgQLZ"
            b"6o3t0Yc7ZvNys6MLL5677IirKuiy44E+qk1A4S1a3FvKqhXMxjVVqKmWh2kzbyof6ZDFcie1xWt5+TiUMvRi+tK64jaO7gzzRuvm"
            b"GNq2zczx1MuyApplBbxhWXEgDPxbVnyLZQW1yELQisk4Q1BDlcIgmQVshfQsaCI49aA4ksSBwVJojSUh2MUdlfoRagfobeozeKkW"
            b"bJICCSHY8UClTzFTtQDLGU0rjERX0v6p22V1pmJJfa8LpjN6WDaSWCHZzyYiPNXRMO0wuvaoshQC8CoAW0kt9OwlGF1iJZlKLAyI"
            b"PMoCgyl+HO0h66+K4SZ3cE8MJo/qbiQ+IuSQHQlHuz1Wg5krYO/UHbtca68F8+1FtxotA7bXha6ou7xRy8BbhEgI1HAeT6YJilN7"
            b"jG4wEl4RZCQSGjvPIMaImDruSYyEhRLRMQZOgv5EPLz6rG2yGgfL55HygzD5AYc0HpFbj0FwhzD5KnP1gZJ6DJW7EMAfe//IwgKN"
            b"z7CDjyLusuxPnhYO6eYjVfRIC+uMYO9myF0UsOIrGZQxxmfIxnnwwDhi+7Qk15bFPouUzXKs3hBX34xUGeFSIHVLSOtcBkZu1Nbr"
            b"+/bdmQzM8p7WH1byMmQjPNqaP9qURunoGLx5B+7vK0LTKlOxBKAXCKw/GoCeB53OYmmNwUISRSlONEg6WBASopsOVmqthXEgNZce"
            b"nALtibfRUpggwD4DdGYnKe55xPkdktszeMuaM86/9VG6+AqceSuatiXTu9jpe5nxJpEtL+Sw2SYtINZu6bqV7wjRntElTZoPRXHU"
            b"i1TYPqYdhTPYaedMIrySpVlh3kvOmQ1lvso+QAjihwsSAxFwIRbjZ6k5gEC6g3daSGh+xvZcgqwg1bLqyJNVzDorcsqTImcd4bZt"
            b"JHQQ9fL9j/wb5ZZv4bEfsCA9yiG/B6mdB7c459UtByQAG2atCyIY5jFSMTCWijNLkuCXw0ZZ6ZwRmDsL1nvjFXWYRKes7P+d8uGH"
            b"cgEd68VJLoA/kophI6nDeS5g1id3VRxRboXBNWuw42qGSoo5qpeXiDayXZqIBlmBEXywzwrc5by4lYWdU/GPSTrPSfrHOYGj/lVL"
            b"yv9D2Cu+ib6hj4Fm8mXcBiOVFFwFgq3A3gcmBDYeiJEsOIm8MNw5JjFEB8ik9iq6xh9XI7sBPru4dn5f1W3AFoHXscGIWP6z8LhF"
            b"ODYzzzX3oASE49t7FSp4VXwLstE1mGK+rMI/x+jayVwXAg8IhmxW7janHNMB0k7MrpTIZR9OBd+SJnyt2pUksTyC9Dh+rTh2T7V7"
            b"Jpwipyxxzyl9r4jpCdlyB5wYRbRva5b+q4E9WQMLMcw1NPp3Y6QFzZS10WdRA4Zp4mSMPATTDiEZI4pgFKfOcRrtmOcuWPylfTYV"
            b"k8Q0xsXvDHMneYSHMqrHVpvTNpSnkiBHJNo6C6Cub+hil09Nmk/yena/kiNxflHkmvDgP0Xx/4Dpuf9qLvE2d404K2PzgSL/3eT4"
            b"t5pjnml0mTUvstVqEzNct+tGmH3rvBHmZ0jN1mmGGs/2Etnyh7K/AYJ2ARnhNXKcMyGtY0wCT33hwgkUtNLG2ARHMEZRK6Nz5pQa"
            b"EgBR/Eyagf93QuXzkursCaEPfqQ6+xadxEkMf6WtZhPB2iSwjqqGM4hbS2g8wGKhCRYr26SrxAu1csPRU+KD1nUXoBuiFRBtPIy7"
            b"5blsVhYRaIReOO6QDPO4gTKKRtmTDt+WrDKUmRHUaNGOrNMlFsOeHn+wqDnQ439cpbCkeneoMXT8G7Ty0Y9lwudHm6U5YC5G+xVM"
            b"Q58Uo3pVaPZDtPghPuqUO2ccCUwlvVlJtZY+REdNFfJMMKc0JVowQbjmPHiIH+PEic8E+iIajysh8YKgPQmJp/GwmkmAtPw7p8Fw"
            b"ffqNCKO4TCRxozgtlxCSdHQ+2RAjpfaTL3uog/mELaTJXqcjJQuMlwaG7ZB77Kjaw15qJRnqn3Q9mPmcIgfuSGxLChidM1uUhkXY"
            b"8kd8YCagzFVPwsDyl3GvPLbzqNGKR40u6eBVghs33k9uFbB9Wy4eE7ZPaJteGKYSanDYdweCjaLTizCwd8SgmzdzknEVl/U+1bSM"
            b"snHFT2KUGQRWhKHAMSdYCSqwROF/7F1ZduO4Dt1QfxCcuZj3wXH/S3gcNJAUKUuO7Uqq0n06bceQLCk2BAJ3cERhLBBi2GDKUVC/"
            b"1OpfavUvtfrd1OonKRBPM6uHMsNvokD8Mqu/Z/sXB2KttkYQTXFgwWEHGDC1xCfsGQOPhHeAuBDWGy65dkIzbRABY9U7ZDdvSAu/"
            b"2Hx7aDxxDUu2lb3nntoblovs1N7jziZmU3Kg4StIaQQ/VrrMMZkGJpnYg2ve3gF2QEubNYHVNvrxdstmX2rStHJ9m8zcwMRDVm1h"
            b"vouEzoy3MxM6hwJmuFb1rKiEx1OVWecqbfClpgLgm7pIe5FcPz6T/5zxoc89pfDB/aOyferUPmtHke/TWGioyusq6JSq/KXGwtuQ"
            b"ZoEzjagxyGMmLUiLvFc2CKUIlcSrEGJiNVoopjkRwVsBzMfq3AANhHxN5egrmhXPCrFXvdY1OV4SKr6Oqbgmf0FGegaHVMsal7jn"
            b"8AspZK1sVQEY7OHbezwpxHFDW2NUkG+M6oH4REJS5B+4HIBqDri2ADw6Qanc0Y3bMQILy3i1ghprOefDL0DilbtxdvsqYWXnschG"
            b"UDeOxh+oEhY/eU1R/jbZ+bmE/EyQfjbEe7i36aNegP746BqH7tOSzlXvpJTcT8tTNDm/kad4JAL9mXpbUBJLZ4UoeKKE8owIZZgX"
            b"oAEroMIjxZjCiCuNuHHeBE2F1oRR5jn84Xr7rfeJeen4ZEV+VNDvWCId/+PY1LnE/xgTNGYwhIwUWwd1stgZyZVhTC+4FfJzfkne"
            b"a6aYYFrg2dXem0r/ID2U4ov0EC51c31cNRukA2kfiI4NA6QPzjuXqCaAyAo/Ry8xQO4l8lIP3NGZHvM/YMr/+EorRh4Gfiv/o0c4"
            b"t42XHyrIX2XpETLjM1L9o7JcpXzLgzeKY2AOYWW5kYh5R7mUgVGhCMMgVeCgcVIg5UYxjp1GWrkfg8x4g9XSAU1x7CVfT+4tWm2A"
            b"VTsDZsAJ7qyiYzT7Y9m3vhZxqFrKbbuFrngMsTTP1xM+Wstl3ewUFzMXrO2WOvggyRzD0kGAZKyRAtp64sc3mWBPhkLU59iT41Sx"
            b"QD3KrYoyLPtm++Bw8qGX4Ccz9T2c8xyZPFeqmL8yA3LUVXWHc+5+16Ocf3B9/YUM/T5EhlFGxiVm0B5Lq2NRHLxiWljMlLY8cB/T"
            b"ghRWxde4sY5ZrmNZ7RDC4Jz5l7FzV/Bus/5Db8RMSx6XmPH2mOTeYVu7K5gVU1FJ5MyTronPfZtM5ohphA+to9ot0ms5NP4UeK9A"
            b"+e503cYPby+HxJ13BiDw1lbnh91Cu+ez202LT8n3QgplANug3ds9ZuuAFEY5Rr09YXda+VBz3Kfgcy+DOI+5gGcQ57mbwAzifO4E"
            b"8Aufe0OytowgZJEhgXEkXDAmxAzOE6okSGYYtZJhDtQjY7ywQRGKTCBYEU8ZfKtkfVZL0/M0fVZFT8F1G8r3QY1NO9O7lIL2PdS1"
            b"6dwX9cQTGvAig/a4Mmb5Ux7/AAuOY239DtEVJ76mj3vi40WDwksTBm9/gbZQbSyoUYmjiNIDLLnJvyXkObc//Bp/lHkWPMuP447z"
            b"OBf2FJDvX7U+b1n6Uhp18NZJyT3BwDmTVDJnrAkMREAQmHNEERcCFsE47rSyCguOkNGSaS7tjxvk3RirvXCWd3EsdsajHgrfXB1L"
            b"TjFz4zHVakF/bap1eaR4a06Zr0C+XKzmEJJ2ijdqHqeDzoZci+JGcyeZDPIyuC69TbwJYHrUvhhvJvLRpQ0uDwzTMW0DwGcGerfk"
            b"LmEKwYCp8Dz8JwfU6UdskN1GsGN/8B4sd2wC37Mh/B3rvWusFzg1RGrLhI9fBRJLXoIM49obBYFizLViFHFsjfAB0zTVQypoox3E"
            b"+8Wn3Kv51zDY5wDs88kaPRnN8Q5x3Zlb8zXlDwB1uEj18qWlyXdN375RmkIEogc98zosvZ5uXa1t9RBVnI2j06cO4X1UVo+++vSV"
            b"AnPrGmQ39RpAui8pkeZd5YD4Wdok8NedNhPKHFDXsXehyelG/QH3al7j13gDYMtPt68ur7arVCv7J691sX4pyrieh83gbDVR8hJm"
            b"+JVG1gYh4MZSwUJAxHviMPUuiESIs8Y7GaRXgnpGFA42CEeISDRlY70m3MPfrFS59fBQY+jJj/TdPKjCrIb7tjOT9CJO/xONUhrc"
            b"hSHkg5Hpf7Je3JOlVGwPjK8EPE7bfvSBTpdDSGbfLd2QFgnWn4wokSSWa9thDOu5HJGPVuAyIasPt3kLfNZ+hrq+P1pfpGu22l6w"
            b"9dLR0YIg7zGFbksb6MrXdu/5sNXinfScMOVtK7219KzL0LWIlP+1BI9r6IPaOE92Nno7DqGwNOiOWJiQPdroX4XKFwIUDBJKepfk"
            b"KJ2jAhtiMTZGKWWkiTmZgXSWKMe1iylFSS25ZiiWo1pKSYF8VKHyPfiEtwsLPwYfpNfwqazlVbmbq42N9F5Zj4FVEphrtu71Lxe4"
            b"bylzNw1fetypYCWsVvBZerBH/XxZQdb4ERh8AKylfaa7BqWk9q6mE+/qHJdtipiAh+qWOagcCTDJxcCpqrnMOSaL/zyZpUvD63qO"
            b"lg12gB7aBjVEjDZIMloJQtT4sFoteIQc6wQieP/wSNr7ub50X8jM75p1GaSxICF+l2O1LJECnnC8PGAgudGLA3YKW61C/CTFtCw1"
            b"N8hwgoKiGjv7d2uoXRBOuCazdjmLH3WA8XDk1usPL1m1A6xOVB2OYOFJlpxJAHdcwIrFcZCXy3BejndNjek55b2l+RYw0nSfoaZ9"
            b"dG3YJZkSxBuVnioDt64hZImNh4QbGsYiBIQHlzpFplN4VlCNgqKpr34nDRcPjaq0rUzgcrG6wHrrx1D4zmXJk9Ji86yoUxC8u87V"
            b"zyb6PL33G9/L6N4e7ldp7WVKazEpx7cyQqYZnHYSOQfMOCqtRaB4QDgggwPzlHAmVCynudJGEs8loQj8P43nfUJAbQQzuobTvYOq"
            b"hQVTiwQbAx36foQouhwg11KU7VLr7KuQ2rzX1Eeli3mnaokMbAySpbnvfQPm+xR0t9wpESUHlEUCj3XaawSX2+vnkL43Z3EzqC9M"
            b"6dCPNdtG9tCzydsos48MPn8hv6+trIOgJimxSi8INUZJKWOhjZCOv3PUCudjxubJhBcLQNxakyfETvEY565BLE4S8txfY5KNySPs"
            b"7glqd9453otDgs8hDMcR2RV/uaXE5WuA2gzU2kJYDBFevANXpaq6RVYtqac+wxSRzl7VhfURgxXPLkUAQhtEYNNyIAfBzq6pvXe0"
            b"23SH+NLWRgLtl1duINg+Pg9uRDdAuwmX/Us11r4JPHaEG7gOgX2dzpoB4hiKi3qqLfGY+2C0YSlhBaI9R9Yj4yWyhAhBcZAIg5U6"
            b"EIuVivkKvQMT9kXqwB3zjJehvgaZbTj/n8OiLtlUzAgOZ0iyG44cH0WIXUR9DUSEKpUfvpOl2SOpH7G9EyOIV3eFgWLE4K8R365s"
            b"x0i8Ek1y33RAex23rACRoluZtZupOH5EXwNlmMHA2BQGxqYwsBqbu7MTKo2eLT2zHQbG9hq03v7HcxoqQbUlnV8QVHtror+KBDOg"
            b"NTI8aIeIMMyKWDxwwbFG3lFlEU8u8jiunpgK0iLNpGEyUKqMYySY8Jf1gtUVh+W+4Tnr316QGps5ZVxp9MpdmWFoarl8yJ/q4S6S"
            b"DGm34r/aAGR4Zhda5vkt8ztXCwS+DQdxG8oW4TdK6rHligLrwktTgyxNDbx67u13neMGIp8aRcAGl230l0ufxWID2vSD14riPaM5"
            b"uaRlWqXldrDWGNA/NLCnyx1rr6plbeo5dF2mAyGeNvKP94DXRkGVtUsP+Fhq1wl86QGPILrLN6e0kY+d4Le2gSEwSHg0Qzw1TiTI"
            b"DbVEO0qCIzwW5VpTjY3TEjQ3JjDppFJaO2Y0E/gNILZJgmbzFMsejNvazkE6qu3g9iNZvqwLojX/IelaD4iRkjuk2/HCKFv9gKrI"
            b"GV3iUopna+asYWKi2jF5RFpjc0JZgW2prXJtKW0pCvf+P0fQRZuBxapm0wShUpwvLYzqHHrpzXSWquB0nzHkvEdaGDPJzl6ZMcnO"
            b"XulJB/eICB8eZ4kFK3OK/2rF2R+MrV4KtMUIG8uRwgG8wc4oLRkE6gKJryReAfeaeoc8g5iZKJXIU4GId4zHSPcPI7sat4kh7v4K"
            b"/KDn3jYnsS2SZaYksH2BW1u30yeQXZeRBZcr3fvu74uDmszWaKjaglUTpj6Jis15Tg1sifrwHFZ2D4zxapPmrQbXMUfTFMEBDheJ"
            b"dmzlHJTxDE8Cve5yHGaCMWyx2aSVzWbdMhjNpNjJTGrWBthKy+a3dUfgFbiCX7hXP5RKyqWCKUMCUUwbbn0sJAOzWhIsgQiGrLVx"
            b"VRS40oFKBQGAYe09mFhb6nf0eF8ymnrQGL6yku/auNct265Mss6avX1DdmCNOWiuDvJl3ertd7uYXjZpb7jfK43wLffuquZQCZvL"
            b"G01aMgLtLmk3C6EzSpc2wQIGW94HFgubDjGbokUBeAHZAV77G463y9FZfv0rujI3TTDkFCErV0IZ1IQy2F+rlvLNY16t5ZsnZwt7"
            b"vq/n+f2l/Tceo71CY+ZNa3tMBXVIa0MUCAYOE0KDBuGEEhzb7GRhKBCHOWEScaOAB2sY5kg6xt5g0bZ8Q77ARDuvfh+bRHRRRzu2"
            b"y9rrD0G6k6r22oSvTmh7ehGduO3UMAIVn6zqkm3W8amm5qo1Oz6czOEO0fQ0ulNKu1vuEwS15JAtE/eyFTkw7z7BwdkuI7Zu0XRQ"
            b"+go7b7KkUmD7OYvhOeeYupl6s/B9kVTNG0Rsnnj0c2vdKYjhcaX76sHWlmAtllZqr5gzIRCMnebSUMEpQUEHBM5qFEjipWmFNSLY"
            b"pUdeSyTtZ4dZfcf0EUdBXjJb6xuXYzTEcDH+XE4ddhrPPB8emtEPE/7l3kMKkQfU175w77Bfm+0FwY+5BAd/CqgaqEeFclI83+Wu"
            b"8gUr0GCTaDj6UpBFFewoMzPs1Z74Umz9aXzZl2LP8ae+FH+kH/xc13feXYaDjvnsd3Tw6Fv2jWcpvPSTH6TwG8OxdxEkLqMWYkVs"
            b"g/Zc8qRqLjEyKHBljKNAQQtFMaAAsZohThOinWXBKxlraREfcvNrS3HNcOJS0+RFthQPehuDXjBaVhSyOYeZyliKS0iN3hzuxJzi"
            b"plMGL/coSS9U6HLrp6w3E1yh37a74OA0UpFd7iyfM6S4r6YzNqRgU0MKNiUbn3WnV5bDQUj3AgTt147ipWoPmDtEwPqgjAaGSEBa"
            b"Cu+U0FyBQcJgFGygWrKkIoZJfAyBcewtCcDog4w8zMcvysZ/SOL8sqHkZXdNXqJqsADZ+VDLnnvoFir6D2LVf+g2Ekfjh5nGbdci"
            b"F1vPQkjZSjZ0FLkcEH+oNckO47aFwlQQnhc1jFVWfcrewzmnL3/1jYq36k600emoUuxXqBTpz/caY7YZx+I5Y4mjNcSWO7du8A+w"
            b"02yzYfnA4hu2maibpa3quefTuLrlu0/anJbYipgJAxaKIS6s8iQQYbhCGhnHJfbKeKYNpTx+OahDhoH3niNgb+o/0D8gOHan7bCU"
            b"oeeG5ReaExcbuQO6xAPMwT26xNyEfiwTW1Wx9SXgJ5Xsq/EaBYYgBNmr+GloPqtiecEokvUJTtAXOaxcbE45ps012S79tGYv23Aq"
            b"OB4cXgeqSOdQQrdexW31nPvNZVol67ZohaporT0faOUtQXtIxSGN3+Va0M4Z+e9uOT/oV3y+GT3oVxAIXGvliEVSswyHE0ZyroTQ"
            b"sQTGSEjGtDeUaeCaWeVkLJTjR5+Alhr/pMb0xYbu1Vby6/vcbT29MRuGJ/NMx7e8/c5p27gF24bXFCyOqL+rPfVbzfJ0yArJQbt8"
            b"dlFS9Eqk+2t61C8yT55bJJ+ZJ88Ml2dGyUNWxoNHv03qTzepSczpRjOnnYjp3Ssw3nJAGpTwJBkqO4wMcdIHy5DmSmNmfPz8Jk9l"
            b"J+DXofNUrucCZu1X0edX0edltOp5cT5T9HkDerpbAtzDTv8q+jwBniZegOXCaB0otVh75zwFEDomcEVCkvmjAVgSw2DSMgnWeR6Q"
            b"JCJ4z/AbMHu1lMJLmdEnVf6MfUJW/klPxhug3/a32GgZSeucrXtia9xCXT5dUxwXFIOyvj/0AYavxC3alLdpgWlPi48bXYJyb2Lj"
            b"HXd7laxEEoJFU+cfdISuUKrTTvJ65zmWXrxcd2d6o+7zOu1r9c3W/DdW+GFThZ/GA2P88Lvwjp/QnvygwuSSviiiWiEkFRVBo6CI"
            b"M5wFJiVIrVRqOASRLH68kB4oJ4QLKYSTmjtnsP1T3A94hIu441V5pVK8hHTohB74epziSQRGJ/SA68bCM5I6lfLuxkuZHuVI/rdy"
            b"9r0u/3tEG4wlfXGFebsq6TvWSz6T9J2d7kDS964/phLAbpWMZ5K+bCrpy04lfdmppO+4TKwkfZuacZP0/Unl4xtAEW8jfFDJGSbO"
            b"Ei8x6CC9YxYcjWWitxZRI7mX0rjgiVfOxkiBacCGa+CcAn+tHOQ91Nn5av0JX8yT9uVBTK1MBJdhGGpP5yQVD3kYxyS7WZ71vIR8"
            b"aBmGprYkx/Yp1RFZdenOsmTvJWI9paNeZXnbFBhzm4AVDCHW5Tl/2mtUrlXq0hev7zV40MRm5UZCCODNQml8APlAc9yzWZXxW6q6"
            b"Y22bVeasQ0os/dRx3QrTurXR1O1sKU49LpeHf5mf5R9yrdxSqHVECiesosmIXSXWJnIursjjF0Zoa40nkhFpscFOcaTjp9ESC4gn"
            b"8bKPak08ki0T88U5PFqcf3EQp0YqNU9pTYij8M7FUd6dwdtlCYnrIIehCsQO0h2pQBC6Kf5sJalatR7pFQgzVJlzAC3JBJFCq6bl"
            b"LDvtsqHaT6mYaeb3YY76wztukE+ixD5raAEE0tW7PiPDKzF5c8CslcjGWN9MZq6Ez+sncpWi5LUWJd+ma3igC9xxngf055/iZDGf"
            b"jn3Jy+JNTVPqMcXAhJZgwTjmlJYkFrTUUa8dBwfOYyY5ZsgA2GQqRCmiiiLpnHL/ctJ+nJIf5/Xj3GtLc4dZVh57URhkxG6ORaHM"
            b"yfa5V58OHw2+VMU9Hm5zmHztip2PJl/HtDk+qnb0VQv6TLc5zr5qwbZrs69qi0uzr+64yOPZ1910/q/ZWfym8ftpnOGgsQ/c8QBG"
            b"BO8YJOkK6uPPgG1yughOoYAc8oFxhHjM9JiZ4AlXgn6pebzqlX3JAF488oB/qNA79DG/oTB/aGV053W9jzDvn/THdyYID0NB+HmP"
            b"4nisj7okZDTOetTbIDvmeWAvf2z2bH//S7L5akS1zh2OMrsrSVqs2RavoIZeoSgFlmvKMKVy2ajCKYCaGEuV+PhztfoQR3zDoa9S"
            b"wlvp+DxXfJ/g8DwzLwpEvMam8YpMMvYDlVOf56ZkH8sUVQ87mSLew97KXeFLjRf5vwNt83+1Dll5vpFVoCersCsK8jVmeaYgv4Bd"
            b"LijIQ6ltZX9vyA4GH4O5MUq0EVQw7zhXUqKkJR8UJyJQ5OJxWUOI8iLeSRyjGtlYgyXgBGDDjLbfCCGBKljpWDt+Lhw/kjReP4l8"
            b"JGu8+CWJXg69rfRSSfAflZWT/ZKhGtk6uToJVZ73g7Lx4HbfdA1uog/SLopKu4JjfU/mYshb87s1y0hXWvBuT6y67Ku2UanFK5Ij"
            b"qjol/agwHVyK3TASJeA9yhVjGO8jFQrcxX1NybgAhd9bB2/JFNNtmbvUwWJR2osX+TlwRP77DGtd9HJwBOOCUYYIJWAMYsgFTKxD"
            b"AVOwQika0xMOQK3imjshvQftnSBKJkF24j/EWR6VwTWGAlZM0rBGfmmBXEk9jIUeNkmJgjw41I6NZEMRCEt///GbVYINeyjePmDn"
            b"HhTjmk+USlcRuX4k65x6LPlSYKoQ2KrpsB3HsJBOcakJjEDu8es3qV8bsBIXs/KuPbncuobxV4nXeYc5pCEt36wcS7/3vu9Q5S1c"
            b"QVrXJfo2equesf9k2x2unrD1DrhWkM2zFbawAMOyctL65Aii/T7VYQ1buE5lfljtvZTKzISNJV2QUmggAeu46OceMy0STcEJxY0j"
            b"RisvmQEwMmiqg0FEcYpo/OU/oVX5QhHKS2KRl6EWlzQluVxRAaWtSVv8LX5Wm/MeXzq/e7rSaalal3R1JjzSpgndbemIrInBarJF"
            b"EcVMwTm1ddXjmJudA/OBPStVGdfw6VLcUQem1RqZHux/9h4sbUUeeN2E5d2o7ej51iyvq6X4eFVeWQHljEt/ZStfiHFgWsXEKQhn"
            b"GNPAhWXay8AE8jZI7a2AYAgFZhEHGZRRgRrjg1QmLrNB/7g+640G6tWWbJF+WbVu15T3SLhmWCQWQRy2CuLUZKp+WnWvzXq5GSzy"
            b"u/d6DEPI773W6CwzD9uW5wadaldVPxptktxSltXxi16jbLA4SN6ahVZ8q7uaz6VswmPJght9iX3zRn2NLKFfaq4mby4m+J0aecnA"
            b"vE7HfEf9yr2n2j5bWQ9NN5bXqN8lL1cldIs0hvU9+Pq4suNoyuhaTKL+fUXB+HYV9T/Vb9Uu/oOYwohLYTUnTjOpkaPgNfI03jS8"
            b"0DJ+gGnQJjCubfxyJBE2jIkPH+q3Pm0I90IZ4hkl7CmFiQmjrSvRp0V1I/7QcjaylmMt/zM70roj0vDXYKOw9TrM19TlxW4lSpkc"
            b"EdlgtPu043yiRCHV3Ku24n14zjm8vFUsBbjo+43dsJWXydyTSAdJpORI3uZt5L/3xtvYnqXOwpJFly9y/SxnWLwQPZY0Xj+9luX7"
            b"BkZWCdpe79V9yvvTv0zg4Q/JOCw5NlXiWgue9NkQF5oFgCTixqSR2jBElMbISaZc8DaBirjgGnEWeHAnTh1/g77wNarcY3HhS6LA"
            b"16l0D0WIHzBGjtrCCSz7TrHgE1bK0ek5XSW5zgEb9sZAhSgdQgomCi/FbXenIMPNUnQ647ZtPLzcOWS5ZTxJ/LhrmzTTFJaLpnAN"
            b"O2ubH7wGGnNaNU14LSvMayOmWWuEdu2QCkv8qyv8Dl1hrp1nBLH0P2QtBWaE0gJpRR2RTvkQlHDIx39cMCJgbkgseJMNKVaAnqfQ"
            b"ldXPV7oi5y2RJzoTbT3L6tHZXTO3upaFNTPwaWnatCyG7wolV8cfokWnkS7t5WwuMZLtBHNZg1Unm0JICl1BzHXXgfXBdEFzAVSs"
            b"tT2dqjY+4ZJzYPy5JuHkLr1ci61r1ZkDLnoLkrPOU6ksOltMBmfL3oFjylaS8i5xOrqIORJIl1XvNibSR+levQs1BaOar816zovy"
            b"Ja+lL/kF7ct5fTse29XkZPq39Rv+XFdhza02BGKF5cCEdCBkvCcrJbSKxa/XlKugEqeOeBs/TwiUZcQmFpG0yBrq/pTX/I7WgkeA"
            b"LXqG2RIT2NbYif4AxRJVzdYBskiKmznRH8O/7EH/wM5TbOv3hwAutX4fxGU3+wN0i7WXforb2t6LH/4Wn8Vt/SsO9DdwW40D/eID"
            b"8DXk1ksd6LmT2FASF+c+ECpFshREjjONTQjGWG95LAYNDYIDMZQ5bBUSiHplBOLox0grvsN34qrHcTdTG0zUDiqNQ1OGqQQjfmLo"
            b"xjInlOE9i1QDovbgmFjobViJ3c6yUoDseFgpLCuEEWiEGltziHajiVajWC98k7/xTg+WVTottOaj0ca55mIFDtpOIO01BwNDDPee"
            b"G110Os0S9yX7Y/5KGtnRp+LslZmffW3fttPFRr/bLd1KsfntDY8b/dumwPyq5fGbKGQCIaQxIchpmdEK3sUiMw2kQnzBY84EUIso"
            b"kxxAeZIEcX3AWlrpSfBvKDHnNOBXK6FfK+suKx10ClwXDDXnGK28kzzdL2SyMzVuWgmF4apUHx5jjikAA9QUjEdofiewUJWz6IpY"
            b"xdh4NL9v1mEgnHYV6UB/IUflpgqmCneV6WHfKSRTmlfVhXkJ/gLFhXvSOI9kbnCVTvGFpDvn7o6kcUYumVXkQzOhD0+omup21hmt"
            b"AWRfYOe+rTMqmKA6CAuexkW8pCgYIlGseKXWngYeC2QESaaWaOWDwFxIa53XxkoeC+S/QtrxrGxEZ4BeVmF6n9NuvOaNSXbIFS1F"
            b"ayU5sa59u/q2hMZ6lKKqSNwWym2RnqNSEouJj3YlHxrX6nLXVCfXYHQtLSO9Uf5KIry1TtWaBPu3TVV4DozJsEzGNpjuskGz7xSz"
            b"FOGwBpfG8+HKpojS8n1W0PF2l5TXGmO86YZOxXH5Mp9a6AtQ6TzO5XF3FcfW+eeo4fhinsOvgmMLthLWIKSYDsQ6IAwRrphHhAXD"
            b"AucJU4USZ0xLoDhYrAhHTnAjPY2fZG2+ua3lW5wbagrYyFO4p4DRW05n42RMVtgVBrQAR1cKwd5glC0aIKc6ueJM5Sh7QYnYwLJi"
            b"lFn5chMSfOTW0AgIJKBqDJMSVSoAswOMZ5AC4x0JNWQycrC9S5+O+D+6fqgA9YM52gz13+aRMLdDW4ZKW4KkW4KcreLPfH87E7Rv"
            b"nwink/gl5XVUsIsT9xeyvSTCMXmh4KWL6cs6DAlR6gPC3Hvg3DjMpPHOU4UsRVorsNQSL7TDgZoPYUrfNfiZwUl7T4Dj8HpfMLfd"
            b"NYLXWU1JHWJV+cCj+UoZdafQZnpdDWPavV+AyM6hr+oxprUb3cxH+7gfj2fOqTpIGnRxah2ji33yvjKx2tgUkkkii6dXCdwQVTlz"
            b"tlukyGW0z/GmDbZdAN7/SVNMrtK3Vfq9SdNNIa05VQumVK3ndG/narX9Qn1B+X/HMdQlKGn5gjyCkr5RWOB0yC4ls9xLIpkQLlnG"
            b"KAtcOao5cGklIMadoga0ZEYbAKqMByo8tThuQn6S9eOf83R8xqcxnQvB+ymzltfUtyvxxvdimDbGwazlJ5G3+TpeJODesn+MgRIN"
            b"zB87vfUUVM77L/J8fA1K4GkswEQrBg4Dq2HS/uNogl/Lx2O7QBoenKGMai8QaObAMwBrmWFIY5o4WZYyZLmkTIeAUpWdegZEe0Sw"
            b"eUdP9vVwgYmr+uVB+O3B9g29xbTHJUsDUXhiJIi/6iNZgQi7vR3brB1auNIvWKi4eGkfLyiKXYKQdZ3udELpJ6+Bu9MTO6j7kk7x"
            b"vDt0udw2+HoZVP0OTY86HUIKZBjqu4waHkZ6/xLICAYyaAF3jN4SVi5MS9C9iU0oH5w78ATeyB/yRpk2C9bs5Xr7rPC6aE3rulK+"
            b"1zOynWtb692OsvtQA/cz9IN3QhgqjOzSF77Ayb0AbfhI3rdOeeutVVopKQPFBLRJpbxm4LThzqFAiGDcExHXBFZ67Bi2xmgC4Rvm"
            b"/alyDX84MBvywa6QEsr3Nf2ULT1j+fu3t4Qyft6yYVO5ttFXkzJf7zIES9J578r9Y/eAXHu8eay+E/J48zgsM+7eF7KsTb4KIICz"
            b"TgkHH65FPoxyfiRRyAeMsuXD0G2Xg/OaiaOaiMZbCkfLK05HVOI/nfufoP2eS43NuWRsDVyet8/YGloZV9SE4JXc0FIeKhGGhw9X"
            b"AvDLqBK/t4EnbwMKe6GNwYYAcZR5ZJ3xKjhMtaJEKePiOsBorTSAYsYI6V38AguhsdCc/NKGL8MvRqC09U5xDZQ27CNdIiXPgHgH"
            b"CvRy15GNh/01JN5QK22CbuObVtwldNuyzrqIbhurr6WD3dBtVK0k47kkXAoi9FmoBrwKCQcV+g0uIeHO+j5b7d5B3YZIuEt9mV+K"
            b"8BNAOMW4wtwYm94TI+OCC1KLIFD8/HMfCLYqvaIROMqto0Q4AEOCtoqh8OPk0y7U0lcr3ht83Ou+Cpe7Ndc4w5ftjw9DAFy18493"
            b"pOMQoK5myZJROweIwRDgqFBG2oss1855BQHEK+JcTjYSWXYtbXBsuNP5Zvla5r/mse8+lU9ji5wmqYzxqkXH8V1K7Nas/5w/xdw4"
            b"bgYWOfetOOv4jMQdNtcJvgoDcTrXyOx857YOz69u2p/QTdPMesasNgyE4NqzmEdZYNg5pZyPdwghAlXeIhWwQ5bGRwI0l05IRCn+"
            b"VpzBKaLvnC34oAfUJefO+XN/zwn779Cx7kQmNmegR2rLGzi7OyReomoOMax5VNTzg/ooUYHnCdj7RKQLb3l3It9440m1re3DVk9I"
            b"syuBV2V2Sg9987YNlEMoWj2Z6EpFZK2naY5oFHvuNsmxvKXCMy6HF6GcbdjZvjIeg569slv7rMWz3Arq9dH373Gcybm/lZN34OFp"
            b"RYNEcfGFjGHOS8OlQZYhi7kwHAHVSgQCwkkRrLbIOqStDzRmyhiM3lEjN7ls7mFxXj0/KJ3fZcY2bGb0X+LrTOtDizjvDkbr7qyJ"
            b"Q2Bt3BaZ2+Urf2ZEXEsT95l+cSDOfQKJDjPTbuf5zUskJNO/x3q7mzgbrPJsQzgKW/aXfi7hi3DwcAxZwnLHghNBjgI7hw1yGKCv"
            b"ieuQW4hoXKGgcYOIxpUwDm5EcnCF46sfs+kg8YCD3qgjfV/5B4r4VliQbbz8kPH81pLzvMxURocgMLGMcuE8YRZpzrkMhmLkhFBG"
            b"eiKQsjymW8s1IcYTCDwkuUl9nzHCpr7Hc/wfez1hBB40htmsMdx4CtWH3Gbt7BY5+FIfS68xexpvFBG0pbi8uzH/AlCJXBklbBEj"
            b"F+8nlOS3TCGEAa3dlsQJtSSFpgYIE7K7mGJ0L2QlkrDK1WIUlwPS3XAtMW/6X9yER88IJDAlkJzJQMx6AmNZiKPz8LfxuJjzR8qA"
            b"d8QfeeRm8VL+iNagKTXBGszAxdusCpJwSyRY5RhDQfKYECnSKn5cArJaY++CxsxLAgT/y4bvD93cXyiIfkNrolv8N1dlSjFJexqe"
            b"5WVI8qByhaoxOxhyia12VgMT+/4KZG4x3yXU9+OGyf0pHfRSIcfKt1q170YWkwNDyxbxA65oe2wwGsGtUzVFn9aM4ALiEoHTW4ZD"
            b"M/O1s3ZrlmGCHSpRW/7K2tfi8EQ2lr3r75beavWw7rj+2ru/Q5tHa4UDECVQLDaIxDhoyVUwkhljgxXWEiyYo5w7jYg02vugVEAk"
            b"5nOCwzcy7sWP8vJJRj7lVwzm9SvQGUlcUxRALvf1ToTsAlPvCLtFlerllkBjxICqwTqSRoKetr3FRf2mIRzSggvoW5w9giGdY+lZ"
            b"MtqMkhafhvZkc1TpZaCBQ33nxlTCVjd4Mo7KL+e72yYEeS8hxrO8R2Ce6TjQBSNQUuH++EzFgVaLd9os5I/L8FbroaTJ757u5lqQ"
            b"b+RiHAh22miJTbxvaggBYUG0lswHpAJ4JqnWNH7QCDZUB0cNx94TzlGa/jOBhP4Zcx14mWTOWNYRPSGqc1WdQW4VGuoEcAaVJytY"
            b"KJQAp3wA1jqsp1sbhRVR8cj4obsosvg91FLjw7o4H1TWGpeE1+TlXmsnvRx/yDUm5/2DMk6OeE6ZIZfydwbn+2K6fpwSDT6YTNbT"
            b"mWO3UU47lH1VB/1g/BNzm68L1HxBleG1Q5vglTZJIFEQHEt74YnViHuOqXPICiaC4s5ZTRxxwopYrHlwWjNGqNPwzcVn3qBtWy1b"
            b"80pLzofWM8Hadmq9jaz5VY/g40ENptb1KHkRv37F1LqQ1MQuG9ZGd7ysFFeyt1j5FmpYg6WX0z1jHVqf4EK7sfXShH/P3Jq/SON7"
            b"hhadDbTPJWnxuK/4OfTnjbF1Y5F72Yf84eD6pZ1Fo4QiJgge/0soTxGs9ZgHowh3RiCb7MmNwpYjT0Py5cJpqK1dXLwE4n8uo/Z9"
            b"dKgSxXKFo5Bsi6yUJujznKi8w5QsEvenXYYOuEh5lwWCqfiRClw+ks8wucr+SnZjCMl+13kvTV2XgnL+TGzsao5UJbr+VJfQG3yo"
            b"E6kHNZRPyBcRK8rIOtRZTnx7I9JdoBxbDukoozAE5x9lFJ6EET1hhHvQujmbcctKMbx+/GjZTjsTxbGQYvUw7bj6bT1X/+VV/Qle"
            b"lTHOGMGT6YyO728VwR5jqhz4+ChZf4FTimplqbMUWYq95pY4SmI0Qm/oVC4oxfv8qW1MPp+RywfatUNs5JSv2/N8SOY1ApdLud8Z"
            b"G8q+F3BQCGuO/3Kb8+Jd76JjbQqIbyeg8+s5Boot6RLE1FFvbGDMeEbXPVzNtFMorCksdgHG1q+8/1vl2PwOybiE7VdXLFe3a3Kk"
            b"GF6ZgjUwsjdl6JmyGJu6OLDlfkarwfy7LMEe+TV8aEC/NgyqZFwapKNkXOZB82S8ZmMsylRzNLx/q1SZBR4YdTiW0JhJEqQI8avD"
            b"pGAOGGZUxbJaUGFjqrGOYafAC3AQNKExHb9JqmxShN9LtjcxSa/LZwddHGgxPDBEOp2iGPOHaKnIF2IQq0FJLXsqv3GJ3CvhYsiw"
            b"+cKQYyWclx67hE61lBjNcs71ceAJR/K0p3SbWu5lqxwmGg3ZctCa5AlvkbHHo01vnsZppHEsnwikkWWXQOP/G4jpvsHwkuT4cgFj"
            b"XY1b/8cOgrLV4ThX3lsdfhdrdS/BL6mb13mcX3J3vF+6Vzm8Gn2NVM5bvXP4nmn/EQygaazMdM1qjtWJrtkjHNdHCnBLmCGKE6St"
            b"wUg6aQmRXkkEQGn8vALRgoNMdwTnEMecWZwb1cwz4eU/dWe4Wolfv4P0DRioCf9d+h60VKpsv3wYn6i479XFBFcyNqicmphyUEsl"
            b"nCOBUSSPCfy4f7U46yQ+LqF0fJfqN8PL/uNPQK1UzniDrK6Tg4GDUFurDO1p/7AkyIGpbwelWFeDe8VgoxhcziTdB8RhUNrdKNKl"
            b"efY+8YRl2whdOyv2YVrsX1VCW+8Qx975mQ7aNSO33zvEm+4QVAILGiDEW4EKxBLjvLHIaRY8A46JllqxeJcIEivlGRMKkxjhnbVY"
            b"/z13iKuNjqv9mlsiyCWCKtwIFM8EYkpcOp07OfTCHfBb3StgT/qoggifbJAD870C4KhuebiPL1cv/Wyl9E+uYtl32YRTwdQR/dcl"
            b"/RyUg7moL3y5RYg2lBen+qfvEHE7fl84k9NaRoHTWjiT01o4c3/2DuHMw22ketjJ4l8Vz/y9bbzptsE0cQFbyxx10muBsTPeUWGT"
            b"v5JEKlZgBnFpnZFUasksJV4JIb0XgaB3KKbNMC5f0Eo7F0obk+HIQvqiQ3Mludkr9Zy1/yjODVtUERsGilxY5izOFwVmviYpcTA3"
            b"SiEC0cMItQ5Lr2fNhHqKO8aIxMPLtTtC+VYmV1sjOhM9S4G5wwRLsq836e6qrITFmys5d4+75t6XdrPoNTyD0UtVwD30sTxYe9SN"
            b"9db4eG2f59YMrV3p9q5LZVKHO8e6PWv2T15LEf5zemUfVCVbsxn3mKKUsrxAipDAGOIkOBIwGGdAeEcIT5wLKYMmRAgXE5tDNGY8"
            b"wuGjRfAXvTrhDHjMRsDjDofLKmWu1whFjplsLUIYalOQm0QNtMqLMbyrNKQLASNroWvd/LyzdEnIZvxEtv4yuyR7iU7smNFCLpGN"
            b"nAIazhWWnaafpBFrQOd+z6Qc+1G1YVjHLqoNop9m3vCkI1zGb1d8y+vwZ4Jrwd722co6q0lq7bO1xqwFxLpXaf3i+oTgMmrf8dB8"
            b"qgl2ePLdOMgXOCCNUOSc8PbOwvO82PRegkYeG42ZlEp6oQxFGAfnKVZGI+GIkAJ0ksMh1sQvp/DcMOuN1PbvKzY3tDRZ8dLN3eCO"
            b"ju21tE02iUbES0ytMthLE4oStzvEsyV8WFhe0+7FhW+3Tk75GkcPoJZ021CXSlqZg1R5X9VWs331m8MW3t0JkDqdsnq68OT3NBfG"
            b"VSdMq84z27kZOGS2CD/+9rfavFhtOmDIIGI1QzoEx7gh3AEXMuY2G0tP65GVzjOriKRMGKoSNYTK+EizgD/E3+31Ec7FEd6jezB3"
            b"w5SP3TC7tHDVvO7aWI2v5DhYWRuwwruWpHiwrkuePhnEJ3g7j5oYse0O9JvsDJnasN0C1uVDoIiKpnAdHkWOKkcQ6z4GRSRn7OtW"
            b"Agh+WgXhru/PQlGrKGztVGvPZrTlgNw28qzak2P58Cpgn2k9NnX4iX5uf8i1bU2eWGqpvTCGaIMVAh2IsDogBh4koVK5mFat0F46"
            b"4QnyyFGHrcBMaUH5371Ufw7/NlmAXysMrzUHbq5/LyLqCvl304SRla3auEVxradwFVlR7CnS9SMr2m27DfDqNqB6YEKOh3hHgN1Y"
            b"p1GlOW5Xoos2DYFNemG/4zSJOIfkP9uzi/X4fmnzm3I0w36mnGqAbao1BK8jpupZyb/ZlGcVYtie1DTlXXJ2rPuN89yq/u3v2vx1"
            b"a3MnNYr1LKWeEEeVYRghRhwhCIOMj7wGA9wroa3SsZpVgCx2QnOhnTfiQxSPl2TiaRruucwFU4vZiet6HsdkORpOGzld0rsw5k9R"
            b"tkiohbKGI5ehzzAZWMCk5fyyOJ8pWU8IJKJf7KZCGBT0Y/B8c3pIhnloj9Gn3Lh9elktY63tDY97yweV4hoBxLvci/uToHGmK+hb"
            b"XNWWrQ7NOMudkTnGSjTVw62J+d3JFqMR+CcoFcfSUrpElHBgqREGGWuNkpZqZ2iI9/oQuDcuVp4WwHEufVLg8tpz7oMlFr+DsvxW"
            b"a6+riKcTY5hJ3bYrdQ8LtytK3dcptwP4FK5UFwYKf910ZT3WoVp1ioprabkJbol9jj1QzM6Hm6JhVcjCRYL2aDhDHmnd4IkoY9l1"
            b"XhwiKTaR2+4dJkP5q+i1dWiGC3ZZrF3UymKnmxKS9S/AMKWrmu7jI8rnUDZpiMzvM6OZEyjOTWdo1WSljXIiVBqL0Ogtbi2CZpA0"
            b"frjRlw8Pa+XF76/Ccx/s1CiCN2Cnx7o9HwA7OYWIJVw4ZMAHhg1XUmBuKWLaSaqltjh+jiFpLErPnIt3ESE1IcQzLdmH9HzOvBrW"
            b"Qc1chPGlTmcHUW21JJmeftaLasMxqChqI7ZOkoZR6fUUpWrJ79IvPgbvaCxAIyOJAcaKVboZ8AhmBRsgq+e7XXObEWidtq3qkr2r"
            b"Q7/fEkmE5Mv13raR46ulSnCj43PTTyGfyF0DmqNC+Fl3YDaTOpNF24XMdgOaKgl/O/OEen1/XcvnoU3CS7V8nHWMu1gIYx1oMCAV"
            b"AyS5s0I4JZmLqZBgYFQybbQx1mumbZBKxJxJPPyCPZ8Ee9KlKkwNzjW4LQxvj9FFqX1VUTjcjqAp7GrjK1IwNf8A6PMudv7J8Tuv"
            b"VbXrZwtbd6s322cdXL598juKf3YU763TJFZqWMuY16ym2ABV3nOJHQNKiESQGLSOxYWd8SCoA6S1ozh+0DH9Gy1or/rFTroHfR9x"
            b"rEMGEx2yubIYrEWY6rW5xsOa/qwO1CZcI6L4API6oDZh1JbLF6hN+3E/pMEeqrxje2VAg+U7umuEOz1ynKAdkx03GpJnVwgXrEO+"
            b"wZvlN8mkNNao2dHFMXe82UK8zRt9yZT2n+bH/jrQvm/9H0hc4DMVtNWAdbwtCO8p8SbEVVfQYCkSOsRM6I0PngbuvEzMAUnAW2Hp"
            b"dwdsXUNZDYdL+MOuNyPEVm8NI6uVPx7V79fwYs/rK8i5jU0Z/6+yZo2Ue12L406qbBMdA7r1QaotzoTO8iYZN9AqlpF1syOA9qhb"
            b"dt+84SXWtGemtV+xpr336Be99Tr0ViDxXwZGUcu0ppKrxAgRSgtI6jMcexwkolYjwW0M45oi4hzySBIW2DtGbFegAeSVBt1nNg3w"
            b"jL32db/aawtpsiNPaTHOqhRNBvmUZahqCgWsdhFz2JNv16/MUamtQNAurj4Q4xqoBh8lgztfjDVPt++Y36hMzFYHXroY58DA/Ecs"
            b"gRCXLmLHaO0btKKQQmxgK6p64d6eb5FjCO3MbG+q9t4decmaRNo/mYjwbugsWYOzZIXNGrnn3EVmvbAv+wmZ3msjrM7U9q1ivOeV"
            b"a8yySEiPsDCCC+5ZiGVsYFgoZAQi2iiksbNKhfiB48zFwjZIi40VGIz5UOW6ZNkOVD6sa5Nc4lldqy7Vteq0rh2C279S19KnmAjT"
            b"unY7vgPIdljXbpfslXVtd0rvr2tpX6Oe1rW0bzEcKtrShLg+7LoFAhuXrWevzMrWs1dG4otXHr23oN3yNKabtepS0IpFqTpe+esF"
            b"bfFBeVTQ5j/nsKBF7y1omWMoMCWFCmAJZwTHJb9GWEkjU761Epg1njkqDMdMS8ep0EgDMvbL0rufVjgXTykhDjkC4xnWGPcvnkIw"
            b"TehTw7kV3/itZBPg2tq9YoCNKkwutc7bYe1wDkZi+f1zbOLQo4NYwM2bzDp87Gn+7WAtv9OU2n/k9ZfcKMhI7rZNpElaFz3NK7ip"
            b"dDvXpR0bmbE1QVbcVVwlX9w6/hxVbseCt91vafXo75Ke+kMCU2syVY47LSw3KNafWKr4PTdCEGW5I0ClVlTHG7mhVsQPrNVaGSqD"
            b"pS5QLbH+xVl9AWeFF5zVciLVTOkAx8qVYTMVGgX+MBCV3DBUdFcDmAGoaImrDdh2FMUBbCVK4JfAVuy+a9oRawVTrNUZ/3+Gtarn"
            b"VJ1r2uvACP8c1io4lry6mVQUMUOYxuCRQAgxDHFpruJHWmPpnZNOBxMf48ApJ5Qb6a0gP46EcAGR8HqeAl0IAJWLdtWGbdD/qSTt"
            b"7buH0P8UtPIQFMFHC7XHBNtJJXudCjFomI6tw/pyF1flLh9oFxC+gDI4HVusHbEAKhOQYjwjsIi2rJdjKJ2yOKfxXDCzzrV8Ulez"
            b"Zedfcky74d37IudKODyro45L/Yd7mz46xxmstSz8GIPfv41aYBFgC8KE4OP/BFYhkQqAegFBao+Ni3WukfGmIJgGzChYqpHT4BmR"
            b"MfX/jen+xT4MN24LF7FkV7PsfIA34HR9rV1xnr9T1EaRk3Iw0TtuswgVoDylE6Tpzp51O3Jwbl1Dt0hoNjvIa5Ujiz8VPhihjo+w"
            b"xOZLwDDFst5s9j7lqEr4szcBlLB79xwzR6wxNmWNsam3+9kru+3O3u9d03/9aE4cbg01f28Gn78ZxCylCJJYMmUd5565gCz2PCjw"
            b"SllleCBCS6TBGU5IDCYsfuM0iZ9n8Kf9Dxh3QPh/Jz2QL8Ep5KnSAv78faVK23XSfpEF0OyOskk1wsza/s5tgu9COmS3lDjhOtMl"
            b"Mv3kDQB4yo4uFgRCVhO5NuO3RXjyjM7LA9pmbjaz5pFsCd7QFWxbFsDQc6EgLOgufcNWNcmRs+dR++am+jeRkqNbjWpS1kMrlaN7"
            b"yoqmTaU6RhpjNijOrrvPQvWMnUrgslqqdqD+sGT0oozb53le5sU/OtO3jZtGEedxRj/vzqAuu9f9HZIfk+pugLcOTsziCpzkCjAi"
            b"qdxgymAmgg9EU+q01Yg4q5UmmIF0moGinlAg3sdKn5h3sOXmKfyz1f/EEvmSNMUkwV6CtPEFc7d1ghsvT3QN/DauQktY2nMCmaGm"
            b"d0xHlgt0CRyB3zodoYJaiz8rtuEw9qLULy/TglWsrDH7RB1cjuW4BgCnZnq/h/w8p/J9OTvHM7gryIOrhjeuxoG0GgfSRpBnZpQ5"
            b"a4XXlphdzX3IzJ1J/bfKv7fYel/Mv6/h8eWDW2rpNf9SHJz18SfTFhSRWiEpvCCSW4jJl3AVE7F2XMQjMzFbBye8ZoErlfgbz+Rf"
            b"doLUmMA0PoXRuCsWzk76p5mn8x+DdWeyVG2rHO2T2uIM1mbGKheuvqIUnneTCYeyyJLREwewGjBH2ms7umOJ0RXm5XW8Z/O9IB84"
            b"YKajwr0u2T+ARPs2QIn7lORHgIhXUpJj+uLaG0asVEIZzY2W1jNuIP4qOB6/TZJxbWJG82CNTQQzQYUAJuO3Dey1jvDJgv4TCgoX"
            b"W7dTObHryll3Vcd6cBUc5B3YMjtsbcjxYHSYgVJCVoGVdhhvlsRkR88OVMP4oKndKNlu3ww4krkutjSOLjmAq3rvMBE82N+I2aHm"
            b"UypWM8+VfLf90Z9BjY05EGdl4hg1NhZjrIq+5eFf1l79Q03UNWeyEDhN/yDsUGBYO084tkCoISAsT4CxgFQwXAjqEAocx0U4oQL5"
            b"+Cn+cTIOZzMt8ZAcNpJpuMSiuqoMcZHgdibcuAtEnN4P2Ei/63hL2P4KI0UMkQ91kNBH8NfJlG64Li6nla8Vq7XWa5cIMjqmB7cO"
            b"sgs4QA9MLm80U2AcL9/zuSwKjFSw1s5sgnUhS+iXBBsgfhtFukbXiWzLepzXi/NVTSeb0G6KOe2zVWixUXocPKv0y6tnu6hYaZrC"
            b"Qbpx4AB2tAlbJ2/fUIbsH5JzsAlORznCXnnMZboPOMlUwBxx5DyVlnuj4z9Ya+AUPDXcY0OoMoEI7p+vsEu2+sp94vwmcT/tv1AP"
            b"aJac5Zvq63u5+EFKpYdDfXATeQDGaJL7kQySu8yMUlQdNp6/QwnNXhcHBsloKVHIHw2/+GZmzp+cu2I6I0VdWFP0rtxYaZoVyx65"
            b"JPP9cSEay6oHKxviBt08d2jvS9YZiV/1Cf+R+fbPZdU1k0oiLOVa6KDjd8BpHOJ/AkAITxhnPFBHBbHexTxKubGWOiERGM2kQeSj"
            b"Zjz3+quPXHfkbeOdi542E3oX7hleoyTZrVM+CKebWfjewLStPLn0gPAmxw2bwceMvjLq8OZfe6xay86BAKj9+16FT97slB84GgrK"
            b"CnC3ydix1a99ArEuwbESB14dHJsbWJbzKBtslfhdW94EhpN3EW0bBIG2eASazXaqERmvsWt0f9o8YWsgX9s29bMROq3rryxlOZ9i"
            b"HGbxfxd9rzIJGoHcPk/sG1XfUqugEGHIaGoNDUJxC8FKr31wXHNHFeFGUM4Jc/H+wUEqzIM2QHHQ77lnkHH7uwKzFar7vHKfS1bw"
            b"B1Js517reOTzNhmsdQoLX/Bal6313Au81vdU+UJbODKaCFwVuDi6x62NlfFFXzQuMnANcejvvNvG/ODWwVfnNkygvSobd0a0eAoC"
            b"a4v9OQU2QlhcMxB8E8RWtVPaZ+y0fcJO3NRXD6NO7f0IXtsK97ZxvjdQuicv8Fn/hDpbPVSsc/mJl9s71dnOU7MjXgUE1AesBKeM"
            b"SukteAGMcCWt81Qno2KKkPUGFNM6HqADQQhm1P8pdTZ4NJp8hFL7MkFlVChfh7AdlHbI2N4tB5AMlGh6AAPJ5TOQsDi0JNhy18F0"
            b"r1U3qjQ6OJyjJXLYjDg2iy+aQs16Q6RvD820LYb2TSUsd6M4bPF846O0obApYTwzp7ypP4yrKSXuFIP3KWWvMjyaXz7UHx5wAWnv"
            b"VfyNReQnE8o3gNLel1oDQjoZaWhDiWIKPNbKcYYQDYE7a4F57BA2GGvpA1AkwTJJNGHWufAMtaNUSW8ohsW8GIZHxTB7XA/LE322"
            b"oaTiQOctfYrIk42aL8gTj8vrXshtP0SYsQEPtJH6ms857MOMOepdiNr6uONnCLyDUThq935W5C6qSMAbunczrdysjfvzzTpuabuY"
            b"ejk/nHcHQixBT9fG+IeLE9Pq0c8VJ/5SAfwm6gbwkDyLFI7pGkvmkHQqsCBVCFgzarANzHpHbZJ4t555m2SJgmBGYZibGb9YgIh+"
            b"I9rdmpynjLqOczAmHBCaIzr1oSPP4TqPo1AymEQbuHdhYwyjL/NSpupDh3y4FvUYycckknScObJWFBqcf9ZeWrSE6O4431bR6aXn"
            b"LJDuqgrPLJD2BsDuXbS7um26wo3I8Pak4kTzxkpzWf13GsMf1xB+BZ1iqVA7JaKLtInXiQ1ZkJp5wCLgEDSRHnGpTFAMrBdBOUPj"
            b"mkjHkPgbEC4ooZEKnmOvIBACH5IHfkkX9qTkvKmuOOtCds5m/0G1I96ZDLML3dd2GZsCFsJAsz9aMXPFlU5y7e7G8uuYVfut5XrR"
            b"BYL0qPGcdliyX8woTZl9CYDS1eN5J3ncuCS1m/WevMeYuC8fuVoG1fZBl2Qq+cnDJa/9xPbmB5uYax5TTDNEiDCKADdJPIFKQ6VA"
            b"8aORxk08FWqaeEcwZzSWcYIHqoWXjmptvpVwwrR8e50DxWF5OWWDrVIJnefChSapXJanfA1Qkw5dfJsUJWFZarZO7Ie3lhmxtTlY"
            b"1vFiv8ne14tcF8lI8BbHNwrOUWldDavdRSOq0B1COtQcSdFqOU/XApe1K9sc0QhF3jSGWCG1t3Le1/QHoBU+qHUPVgN13tip8133"
            b"IENEZD0MWp41g/sS+H7SxNftIU6FJC/YQDy/rj2uZbUWgVvNuUGMWeGNw8QQsICpjQtWBdYKCNwQrpyUoIwn2GrPmFYA7Gcrkz/X"
            b"5vsKLupkCXlu0TAgnV6cYF90ZBs4OZwYhpFqeITxwPumh2EVnTCcp+IKKm+dzcNhpBiQQlfvh9ViUtU14sjJLAbm6Qymol9THzTa"
            b"U0xpRz4nVP4qrcfZK3OtxzMVSHig5jh+9KtR/jrIK9gk24KdU4gpa9JnGJuYZkGbAJSlz51mMbMyRClGLPgQ4u3cGIm1VP67twjx"
            b"ozKTP6w0DwSrtkWIDxXdsKkn1l3BlabeMHqSkQdsuosl4mlT79grPGvqQZtxc0QS9NxuDMvae6DkPu6mNjuUmbGLuy7hWkqe9Qrv"
            b"lpvpqt1bXo87huykY8hOOobspGPIqo7hQaDlIobo02VlkwgvK5Q/LCxf2zS0WGohKPIaO+Uc9ppxQhAjDDMkOVac+bjsDkRY5oJw"
            b"AhFmKRGEkVht/j2g//cg5lMyArRSgxqNppaMS1FeMCGxE/BLMDmkmBy0DnUJP6KHDsTcnA4xFHL/coX4yMwxx6TrhPHi9FBr//FD"
            b"Vs5h5TIwIAr3QPfB4ZRDLtFHguq+7R/lYyx02PhTjBx32sMrYengOItfmpHpTrdBDos7exblD/wWNmmOJpohkJ6rgfHYP2KKV2of"
            b"tb7qv1D+T0P5MeIoIO+0MDHNgw/ecIECoR7HW4KilBlEXACiAbF4H/CGeoglM6ZaUXD/hPnZLfeGh+5nN6zUXtcBSSyr3MuVfWbr"
            b"vdkGk6yxKs1AbQbuqs0UTdr+cufLXNRmGEew97hHfmslJJ1YMjZTzcerc0SPr9KnmwkvU7BZhty1LMFDBRtaqdbQmYLNGH3/pILN"
            b"b1fhblcBK+AMO0DBmuCsFi7e563xSkhqqPFYUAzOGRQCky7pGVINjlLssEDW/mWkKPXY33xYPvfCM/sYri0oWV4nMkwPTgPwUPJm"
            b"6NQ7mbUfVHsu0awY3luwEnpN7WNFfUm6J+9J5tGZwlWTdscOHCv1bSJG9zawWMQZcSehk6MIIVsfawvsehMppBwEcFwEH8W6YDha"
            b"GpeYzNNq0J5rb+Zyn4JjoHdaumd6MKtuAa50C6q+BRS578ozndRpmeDKNZ3gqhkil7yy26ptg69Kw2Atja8qG3wIJ7Am3WowhpYv"
            b"3sy590GJu3wqSz4/4gjeCtfHKlaqMfOCMEJon8RkUYiVrArSMq+FjZ9ghzl1RMccDEYxazRBzEkWGNJ/dz6+yD69mIzn6P1nbwNX"
            b"EmJ6NVW61YitfF26zkMZhqcaWvynDgzRY4IfMLlWedzm3dXC5iJ0MckRy4QNHVImK/eNve29SYAtf7cuPGf55fZRidsuMwE1SvX5"
            b"MEp4zLgFuNae7yw754+j/GJ2ZnCLqCore8r68XkRDJXSzP64vCKrlCwHmt6tlu2uJkBvN5R/s/Ez2dhi54nlhvGYmTnSDhyRDvn4"
            b"kcXCE8ksMsFSg+NhJTtMaRxzwSX+qubyx8k7XuGgXleAvCzbeE0y7Ja64kVHtV0IEtgF+BWuZANQOc4VjncUtjwo5aq5bstQBHfX"
            b"Nj+OAE50cJtMeBARyP4P692ntCcqk7TDSeTYVUNXwMFXbeuei4NQaIwulzX1kBe7HTWn4ZYo+UVhRxKXq0LgO2SCBVa2YcXaZ3h5"
            b"VgvE7E/liSZN1mwkZYNVGLJ+KjcdMblpi8ndMkfWVjqyFppphB7L768xs361Hd/XkiYoEE6cScAMZTln8V5gXKInGhsQMOckDcxY"
            b"AiYgwQVnXnow8V8qhOd/hYTBZeWBU+cx9bSiwMskFMZGmw/hbTP5BLrU3IC3CSIf3zQeuaYNJBTOXdBawyBVrg0QzBtj6YHWAl6i"
            b"0v0e/1clFOgB3en1ZwXWSR4Q38nVjfBu90TWjZP+Sa0FCa0+L7QEsmqrTWe3eri1Q2TdJmmS8q+QwetrccIAsKckxK+SthQRor2W"
            b"XEtBkRRWAAftiQgMMeoo0zQghTAO2gSDvfmrxHMvKN6+UGD3uiLtn5PZrUW1+H5R5B9Vz03iAkVBl3FAe+qHTZVR9rqSOfBLCrqp"
            b"fr4jbD6X0JUnErpyKqErpxK6fZl7SJu1rvk69ZO/ErrvmfwRbrA13oEkihlCgrZaK6oE1RJh5Z2R1uiALdFYSGeDQkZ4hoLhRir4"
            b"KJrui5UsnFWy7KoY10MQw6iSHOImyKXK+aJi4i6n8gTGbB20kXOAyGXJl6oFTLjcD3Iz6IGBpORBDnGj4A2vVT7cdCxEUrZaFvFG"
            b"gpce/w4pNr0HRcCa4yIHjZj2D5OuWz6ZpzEYTzDjHooeTmeEZ7KHZzPCxqy3eliVuDMN3E1N8bsDM6bl7lwR5p3AjPNyVxgmidUS"
            b"pHdUBouRYdhZQbhK3kJcYyzAII6UsMRiK5wMXGsL2nnB/rCZ7xepGzdX6esavSvHHzUlxCUtluPRjcRjpx3joXPkYW7Y5OFZFs6v"
            b"566DVGsjdi5qXvRic2cAbywQwDN9mRyV2y1VI6ECU583EubWmflgt1YCVaXsOSMNpiBCuyL4Jj1E3rXqnXmwQWXiC5WJL63wxPSy"
            b"weXIt20f7O2PdivfRzy6b8MTmUKJSyuB1K1bcoM/8iavXqKJIk7hwOPnGHvtkdEgA2bYeme1UYhaQnWI6zGtLA7egJGeEhMS5ti9"
            b"o137VgP059oBr/drGItWXRvWjbuyxxTyaKh3pxktH0nm3qU0L43fmXFbnpelE8B4JPDAjgVqjiwmDlvPWLVkmO6w0oUuwQwBZnUf"
            b"ZmZYXALzWTOCgTymZ5ewcuLNSO/fYUIPmSOX2NE/0onzmO0bHMfnPTpHVbVmxjHJqRVMeUV0IBhh6zQNAvmA46ExijwX8WYQ624b"
            b"v70IPDbceY0k/vXr/Bl+nUecde9eeaA33+k1Y7nRscsYja8nyEZHs9hxpuDdYufYeSC91NEBKMi29NBjCrMXz23fToKXA/ohtp04"
            b"rnXTx+D6bWHu2gmnrp1w2oG5gtKG9T34+rg2iNv9JBoPueb3LxZG/0V2PH3PcFJKFO8N3nuCAudCgPVYB6+wl0G6gIBb6ZzxLH7T"
            b"A5EJIKhBKI+peKYTw0765xMC4qfYh+XT12KX73fFt4YHa2gbR7P5xFOQG75NViC6vuiX+ca0pe+OlI56UEnaFT7QxnsCosJl1YRo"
            b"H9rJstEcAwjJc5pgPuUcRlHJ1lu13tyLZbq4abC9y310AenKrwoab5UYOpS/zeOXvvJT5ISm+rqjDvOlPvJrGh1rxjKEMEGcYpaK"
            b"AMjw4JLfGQrMa0N4rAohxHqHKhqIMl5i51nMccQFjQn5w73jt/ZBXmyhs87ZKlHcmY7DAogY+GR22jqN8kbTSxalA3IhxWW4QhHC"
            b"FXwguvGctnkKSZd3TYc13JgfzeTR3v5QCg/9fbqjJvkNVObobfPDs8KSLYEJbrw0i0U1QewAyeS/Hlzxzpx51PJdH7etidfKWHwt"
            b"j34bofI3INPe1kumWFPqgKoAGjy2ynoTUFahcJQqQRiQWBcaasEzFYj2JMS0HDMv5zLgv9tY8oyNh6566TwHlbjEFiwjspKlaGkY"
            b"VO2CpWkhut5CCY2ZiqKqhFuDuxZMjko1J0Fob0iUDzU6VdUcTMcm3ZdWLS69T2Z0odWvrZrP9e/KxBIYP6RCNI3dZYPWIVKI4itE"
            b"AOprOvIITiH52j/rmHNX5W3B/MK+Kr+iU7EAIWSNiniOpDe2C/6GNL2/yDuSUsE0k1oYTwOPydVpFoiIy28qvTXEQ/xdSAZnYDBm"
            b"zgjPOTaSxUQiuPuQLuao81uj2mDNSsNc/Nqe8F6RDuvRvaYFNOoaLjroQOQaU8zLh2+W0EUpY8m1vahm8pKJILFXtkW6p+NElwhK"
            b"10J5e2vY3rrT7aRl/a4YHWxy7NSmuJSpEcg9fv2q9Ymdlbh4m6mwumQZ1g3iL6t9Irz0RxqBzLtMuPhGqV6/U7XKHfvbPdn00yu1"
            b"if3ZXCYT1h5Oo/FO61f3hmflygt70vyG3c+6IL0ulvmwm/lSsUzKAqaWC+IFCAhO4JghpeZCMEchKAgInKDcCYZV8NRTY5kx2Css"
            b"rYFfgMNDgMNFe1m+zt8IlgS35yj2AemWL+UCGSaJcH5YZpc1ErkPnriJXMAVcmFhOovpQKzoqedIiLdVeWw4HPevFm5ZGu0RSsfd"
            b"gcNfZglOP2UDjzjZKDOf82ElJXexIuf4Lsw26/6i5Y0SWGJ5u8ZPiZ0oxotlqyexD+mjfJfDQSsKB63MImmlorkDg2nl8Fs/llMV"
            b"uMbzbJtOya35MDaOXIdhchOuWB/94h8+jn+gFmwsgxm2QjtgJhDGdQguYEc1UYGAdCYgplSMEULIEMA5LIIgGCv9r5Ho7uICcts1"
            b"s3dJ03UdT/hLWNFdPogjl/oX37zDvZJ/lw8p3y6PSR3WW3TTcMlhqX5mVPSogMEfqoTltcTI0J0fuXEI1Z3eltrysNd7u4L+jHYm"
            b"rXoStMIXy0o7U97Rzqx+W2ln/rLo3sKioy7E9CiRRCJmTxSIos7xREg2UjHEg8ABpMXUaEdxUqsHF8Dy5L0Owb0BHDDvBx+awV80"
            b"SL/oybi3Neeo/xsSa2OqRC+MOaZKiJEm2qHzPLNwH/E+YN2i50ynN07CD4D3Kb+ovjl9Yzfvk9NG+YGsyDL2kKA4NIqfyUNsWDV5"
            b"gdWx49NagQhSCUQ809u9mV7PSBq0ImnQSyQNuqg+rOkVHqTXKUmjsc/8Nm3d4n3+YKz2osbu+8ZqAXGEQTCHEpLBEE1ocEYQSpBF"
            b"Kq6TISXe+B/X2nvCgVhtkummodz5+43drIV1M62y11uiP+sK0vZ1j1YROydYDMnAx+7IOAMf0K+bBtjQYhxQiYx/LrRO6EtPdRx/"
            b"6BiXcpMeFC/7jvFDZAPMkQ0Z0ZrMQFdmdcZvzQ4wnkgKjDct1F3vFpuQA57FaKU74l3+Q+uWsbZuxzVmjuJ1yVq3bWnlZ1Q/hsrb"
            b"qJYOvtOu/S5grSXpjczQH8O1XtiqZeCDloYI7OKiWzLOLZI+LsqtU9pgqkFjAhDX7RRLHJRHEiPuOGUKcR4+ZIa+JK6ddDUvJdPX"
            b"5jI+YNJs7cf0J24Wh4LzoU36UZUBVQ4QXxaCWKqxzvR87mbRgTtTRALz4wceEvlSpDCIbwwd7vQAc0ghGYUAhG2xbP2K4eqvtm6R"
            b"AnNpAgw2ufSlzTqMZ9u4vxdYh0338oHGernZ3MEH3GqTzhqbsyW6nNadcmpv0bRJO92w48NK6OYF9haPq8htlY/ptuxbEqfYpY9O"
            b"7S1KdbmkMrroQM/UfffkOq4h0VuX54wRgxL4FSNJiYt1COOGExGC80Z47IS0AQKPVSOYIK1XPtBgqKVKCiV/LePmdeDLnM4G7C6o"
            b"hl7Hte9la7RRo3MoWyNKu7DiT4lGT2ZQLc86w0M1hnJm6S+0D7sO1nOHYVcOzSnuqmFdics3pJ1pVm0ykWgY0cCOWx0UJs5oYENz"
            b"vAEN7K6WTjq1O35yM20zOJGEPK1+m2eyma1tFXDF9Bo/3JJ+Y66xw3TLzOzXV+7TvnJMeCFAgqXOGk0DFVZypLSkQfF4U5DANMJW"
            b"+KCkjXV6SG5IBBAA5TKg05YDjJsOq374Wd/hOdXfD7stfwF/wa/KCIsJYpb/N/N4hgUasFXSzWWZq5vh0bJhrhRxvB4jeUlZ2Vn0"
            b"zZshDHkFcY/v6SJPzzISmYDitaDvXOcnR5LFoJ7gZqMxjFcskZV8D/9vPAB9iXYP8L9I9+H7KPfM5m21DUczb/uacs+jTgjqcnfd"
            b"SyH5MalyPa66JQqr+E+gHNMQlDHMKyHic0oUkTYJrCPGQErrqBeATQyj2irstYn/PA9hGNEp7kHazvFsL0Sg3bX0uFPLXkbRjbNq"
            b"jzLIEY/RCMcKvVE44NdgGMME9zBTN3Mvwhd0Hq84cJPMmd8/Rz4GOMgK4PCUlDpK/tZ34ApjyNgZXGHmPvcIrrCLStLPgRW+H0Ds"
            b"D8HAlrzJEdUaKaSU9IbHTMilCIiRZGwUP1rYGK09daDA4SQE7LwG5gJWgir0cJz2o0nBY5HGAxr0pczhi5qPp1qTh8lciilpHNXK"
            b"N6MhXgopcAO+k9KmHZcclYvfgTzkRSDBhCRWMASZWcYrstq4wMxBKbUTsiLSxtIRBQCSo1JChZ0brLp2V5mDfYgT/D10FH45wW/n"
            b"BHNBlLDgVEylRjrrtJfEW8p4zL6WEW2FFUJqQ0QMCAIrj7yIyycjpKX6R1Spn6dcDBbgo+o5yy41/IDm1N4iOvlmZcgSV4gWjNZm"
            b"EruP0Iz5kIGwINTBvO2Mv6D+K56c9OjcNlIFyoFLV+Gp8vWnL/r/YmHHP1y3xgU+QooGzLFQ3DouFRPAubfGCQHIO8zBubjm55YK"
            b"6bWSVnjEPUNOwo+TbLyqw3hd2XHGdBC3HYNuCzHOCAnkIieB3G0EzKQO7zE5RvO6mpw2pD8M53WHrRpKx3Be91AZ8qfJNt50WZ4P"
            b"7djJ0I5Nh3ZsOrSboX3HD4/9iaP4A/xqNP4RjcZ4MyBeBCoxD9wxwYWCBAaWTrr4xGCpGSjKOeLCeu68QMTEigrJEKRWf5lGY430"
            b"ldXbVoDc/yhevk9szU0DVsYCkyvWPLINfLAYGJb5vDSFVx40zBut6T0zExohvFpGpHhxuSs9hreMLDtne0RL4I7zXe9gncDbAvOF"
            b"/a8v2ht6nug/hwO+CXGDxf5nxzi0TGComMC1AebIMEgufGHYMnENi4MJmO2qBeavZGMt2SiQ8FIE48FzxYx2yHBLCDGGAbKJo8tt"
            b"8MxCKm6V00QgTomHgJKRm32HXMMXAQEvsImYj/tPHX+uSJRn2aulIGSU7S1Yvh7jUSE8E3XR6v7YyDTiJ1EC5c3zToEwyXZxAllD"
            b"reOxtO9xQzk9JxAMUEtPiPVYcCdstlb3fGuf4OoUxFFgJ0cW9wmJ6NhmqN8kBeYDirdk4G0zuT/RclVKYCxCEawAhDmXsITFz0lb"
            b"tN5FIeB7aXfcd5BTDtqCEt6Sa/sKrV6h1SsjWYVdfmH/3QpD6B/V4gw/GJpQVarLd/VCpXoBsvCBSlVALG3if8ET8A4Lz2OKl5Rq"
            b"6knK+4ZpRmWg8Q4AQQcagkVUewrKGezIz9Pludg/vdB9eL3x0B3rn8sosRSy5lFVoFwt/GE4u7vYKD/jwXTX4RTXMODpkQUXRhYl"
            b"YtUccDW+G3Q+0nnm7dgBiTaBRqTDzz0WgtiOzpMnYGO27PxLt4E7hfctdvLM9G02qWPTvvPDvU0f9bZxx0c1h/lXfufj8jsCg4vJ"
            b"ifPAeUIPO+sAESMo8p5Q6Z2QAoxRnHGVuhfgIMRKiXoTrLXuGXxx4V68QUBYzAWE4ZGA8H2tiUYcVw0VFA5I3/oIL3NWLitRHErr"
            b"XRt5+aS2GNpl9QDVAa4riLY+HiA+0FxoYgz6QKsu2rFNnY83XWyMy1KmMqpoWLANVCQHA2W0eQ86XCrlqIz/YAtKY7sZDKLTJSmB"
            b"PcB4cBs4AIxvik6wRKG9VdgXuMmajJtnhTaYWZVrrV89IQ2jun0mi+/y3q0mtTJbJYrZEAib31dimXK1GXpZ5+UTGsMtjbDO7l9S"
            b"GX4T6lgQZ0kgHCuPA9bUKaV4AIqF1kRxA1ZLigQy8RPKCbhYBMZKP26TGtAEfRA99x0GludzyHVcN5IWHnVxjxZ0jyw1+jw/dJ/j"
            b"FQaD4SvDzEb4WD6SPs56xbh0z+vg7a9w3IRtoECxCkvISoz+MAIuSxK5zDPrhlF/simonCes6wFcne/WShuchkDLNglOt2xZz1uP"
            b"f6B8/CX6Swps6UN1VyMIrzru1WO2SLrX9L7HU0P6X52hcfXKLv2+V9xHkffvLsA2VTGuquulKXeBEHJB3fhNUDtBAkeIBOeF8SyB"
            b"RFTQAlOhLHijKVWUem1YfIYwcjJW25oJjJR3OGjzjtQ8ycsfkb8cZEd6O8VOE9EwCyXvRLSnFFU1GeRcdYN3qfBu/k77SQEKrxgI"
            b"XktZtO8djy/FNQZtu67yIXctBm2ssxRqHdgyVGGxaKsAcnzu0/aUDCUXsZiAu0ruY5LybAiYonjJUEsRWj9LkUuJuXw162eby2XJ"
            b"NN2zyt3y5+i4jzLgZ/Pcmts40KCdVo4gCI4TFJilDBsChnHFJcKKWiKDir+UEvngJHNgjaSCuV8nt6s0i0voha0HcdrInkhPjnUh"
            b"rrJFjnKSa8IZtlf/dtbISH7yPl3uVkallVcbbXzbxoCLNYpWk782I9OKYVeLSOy6ELQXlugsgevIX/7Ga4tKrYVTTDKjKHLOKiR8"
            b"AAgx03IcbFzZe8wFdVx5TSjWjIrgqAvCGOME/tdYxsfS7ipO7M0q5jd4w5mCW9LmrqeW1So3mu8RTtaVvhuguCebXBzvxX0tM7sV"
            b"nwu4wvMe2MhpPc/TMgxwW9YeDjWfUo57NmO+Sg4dr5pqyyuXLNom+bdZd4+X4N1q/C+egP1ZnoawjDpLPAneesOd5QbHVTnDWBgZ"
            b"HDZcYaK5ACXj4hxRxNLYixChrYv//s2l6ivthC9WvXc7pDO47rRDOq74Bh1SOodUHDuk9Igxbjc575AeroNYWgQkn2X1HouqMBme"
            b"BV9bpXgzId5OvE3x7Ybpbco2k1YpHfGUh63SG3mZQLoid7Bq4y6pnHZJ5dIlXWgYzZMF4cbrQpdXuprygFKr1DHHOpjfTv3h76ho"
            b"vRSeSiuE8oxx4NbyIC0lLDiiPDaeE8QIUZaQANhJBTFFG5LU1ZV4iyFcbYA5kyojj6TKHsCS21HVRdjsGYT3udnXxbx9SQXsghlx"
            b"hb/thvkjM+KBkuQAtvcc0HrqRqxWWXe46Ea8bHB0I14chnfsgJwgRF4iT/bveRJ/EzjwNdDXfZ2yN/kSi8CcosxKzywxQjruCFM2"
            b"WB+4lkKaBCQAihwQ55AUhHnJNI7bBPAC/T3Sw1ck1JckKvE6nqn6pR27IoWQFEp3q4YKm9oFU1Ry9EUtMFhN2eJP0cmvTzW+xH9F"
            b"a0xdwb2msNLljB/HDks1UchJgVnjsTiuq/pmO98Icjcy/l9VKsCk5n7PNlQLhBorhsi6MRnggMkzunAHWDKcwpKPimubvv0Mjpwj"
            b"F89VjnauIZsffDnVEv2sNvFNx465xvAMeHCuWVzco3beNG/UjiuZ4V5xmNdTu765XDBntB/lbVC0X43iT2sUS/CaEKsQkpQRycDj"
            b"WKJ7YkBLwwxh1AtuvNNAAw2MBZDcKaawjNEI/ak6Hh61VR71VNSjdjMZtCNeK9k2QjrAwdbiQof72OYli0smrEgHcd9petAOrsQm"
            b"j6yNsZ3ddE7Y2ZdUPsiDNtWy1/STjFr5s0kkWXrUjK+0kN38ZPA2OSwd0pPNE0IYo5TcQpml4mPHRrTPSgnOq+5I/WzXr+BNBu9e"
            b"pfWL634rvO/M+LPCB3dPPlPif73f/YZ+ytvqe6kQcB8/gIpRhBxjRpukO0SVCVZIBcpRrxgD6qXlTlKMWCDCGw4Ba//jiHxXp4Wv"
            b"E4W7QeS7yjK8Q/i7KBC3SyLBZtx5kuSyJts6ZVw7JxvxfKdiNPS6tOesvYQxamzvZ434Ell42ZufqDoVfSuB+S+SWRLtNuPVQglc"
            b"lIho1dzZzuOoQ0ToEh1/ik3Dk+93EzYaXsYrVeKfZPPxW+LIc3vQGdyDLaLHu3JG+wrtyN14NJus55gjQt8hstMh4r+kvj9A6lMa"
            b"uUBdSOLKgWgXJPaKBq08lfG4HGFSOiBUOomc1B7ZIJQR1jlFnXmHfR/9nNTQVU72ddpyCSmlY0Vs5EOZ3+fs+YZ3m6Mn4AzC0q0D"
            b"YA77GMA54C6cY3xLKDHxB99VkFcrwdElVVCCU8Us1kulRu6DJSK3qDgRrJ1BtHdnUbrbS+X9biunMZ4Epnr1MG2vw9S7b2bjdEQx"
            b"dzZOL/Du+37tkD/U9FjzqpFccAI6SXoiGasgR6REmiDkjTFUK0ykcl4ybBGRFgwCIbmNqZeL8MNtUV/oO3rwt6dDH9ZGNLn82dXB"
            b"5JOv3s+wDhthvUFPmiFlppnwF60f61esYMd1eX95KnDJUl1vLLqh1etUIlRslJaujs9XgSikGhAiWSEmwwuSw9NBUSTU8E0GflhC"
            b"LQf1rFFq4T0/M7bkzdyS7xg+vjef22fFnLAQSJbpZfUsvyr3JF0/aRobY2Bf1/kotW/pXf86p740+wYOsXb1VKgkV0EN9UFgEQgP"
            b"BFNEIMSfliIDCIF10niHGeXBJnkjLv+KNvPr+8cD9ckBLSQHpIYG4MY0qOeqPcDnHekYDK90jFK9NvLEB5Z1fvv4A9pBaB9Il/2l"
            b"n7TpToz3m8M6TkjVmDhwQlg+yiNcBR1QJummubFH1kHsgeCXTmjnjmxKQi/2Brml21lqTdrZUeOtCp29Akej6n1OWPeRm72Ny9vj"
            b"dPDF1L3fBnLbN9DEYMmCAaaVY0YpLgW1Mrgg4kKQe0KCEpxpixRlwLxg3FElY6jFhpqfpwT3cbuQGw3ky24g+8wMrbYdYjfuIM83"
            b"hktYPtCKW1f7PT+8nZw6jhzuKVvFSWbeIeVQqFKkodnM6YHpouQ3uO5RUvZe4jkjtBKLwK03Nj98okp8qv4YRWLfcPYJLGHpU/pk"
            b"F1n+RYZP/4BxyU/oHmvq1f/Z+64Ex3Vd2wmdDyYwDOZ+MM5/CJdBgVGWXKGre9d979S225Asq1wgCKxAGPPSK0YQlx6MY0Q7b7yw"
            b"iHhvYw5xSBmsjfQSEWeciosGZt4HKV7QZha69erKbXrtg/LYZ7qe/nwSmrCitkB7yaP2/QQjUTNUYNOMELcZ5ou8R/euCClJDzYa"
            b"ntjRaWJoNWfgtNyFK+SOhUZtfk8RRy9CzArf/JbZ7xQwO24L2QvvQ42uOSaFJogc7PO3406Kk9/QcG5SJIWqzzuLywFphX9T6v6h"
            b"TwhuONX1wA1X3qS48Sldu56u1X52VeVrtZ+fL3S/Ads6uYubUveoE0tDW1a8llurK+FDSs0AEk5LnCZlzqVmAaKMO45dwLHqtUZy"
            b"STQ4qoATZQQlljAQhHpmKCM/UOr+I5Utfwtl8UA1/32fu6m4/quqtnv3R2L41wVtj/K9ttAj45W/glGMim93YBQTfEcJBoQJtHXz"
            b"/JASmD81xHxOJ6yXHneRw8oH/5jK/d9uu8cmj76xov0VuZ8WtQY8RmkyF1T8W0Ca8fgHoSTiFjmiA5YKx7QfrMTCe2Zjuo/fZSel"
            b"UVQ5/ZbO8b4Nv6ps3+s2v1ffvtsleYJQe08z/yWCrsbPsfVKMl9H7oP7+mR/ZZZauheYbLCI647CQ1xzDiuiRvWKdoGuK5GFmDJv"
            b"Yo+fIV17iY6rEGYn7m1HvQ2+rRyz95vPj+rolUL9lQ796pUrVft3FOv/6qbEh9rOX6RkbAQnIv6BCq1xrFNAMC6Uwto4hRXTHAtr"
            b"DXaMETCgPWIIo6BxfKQCsf5reImvxOblWmyevBKbf6U0f2Pi1yEQboAzGr36zEee2OuV7MwwQqPOe2v3lEJk+o+cSby3jklzj7sR"
            b"wYC3rIqkagw4xsj8vnmrIUVr7oGHt0/IuELlBF79EsRgN5UD0q9CUsl6Hfr206d3zVHDcG+U00i2Ivtwr2OEowqg80l68vwZhG1u"
            b"63Qk3epZDVrbn7FJVCvsWUHYbtTKvdjcT1eMr4d7dcr9kF78Fw33jASsBKGWMIE5ksZ7RFSa7wliYv0gmSchFsg+xALZSOqDJl4Y"
            b"C9QQRcOvGtJt+MUNzaQ/pIb0UKvoy7WT/hNqSLkx9lTYcxRDwksxJHwhhoQvxJDwoc5RZ+gKbjwHFv+KIX2FGJKxgUtOrWI2mCQe"
            b"j421RAoIVnMpudNIYWKo1IxJhhzoWCkHF5SVNH6v/2U7j5seHLyMsAQZvSdaRR9RSuF2ioW3PXITmsdYOWNS0Wh1lPzS6RO99BJ5"
            b"mfWhvBE9QGCicdsY3/S22hPfhKYEr34VUzfUFJJ+HwKxasC459PebjVF7WuQUmSA8I2XnD9cCU6EDdoJ2ncLZY7A6GOuHQ97yiUH"
            b"nlVsW9+2HeZX3ebVSHDIuBfCymd9/AnCyr+GHZPkazkixgDSXjlEwWFmmfbOcGOVDaCYZxoSWykZiWmmmSI0MCoDilU0hl9ixzNi"
            b"xz1WxQ/gfzxnbNxkgiQHo11VD+Q5XmsUlPBw+o9RPPJbxUTKxbAznkD5Uhjmb/M7MGUEC/HIs/rK8UNeOn7kWFIJW3RP8+vyZIS0"
            b"zxpPu0oMdG9HlNdPKVBWWdz9cjw+k+NhYwkca9tgnZCUKfA6FsBGBmIpssoj4aSQiiCESEglsqPEspiYHfGGKvZdzknsm/TwXzgn"
            b"zRoN7f49uTiSajvOFyoUbKNh1Hv324L582naYmvf7OvfUq5PZ0zkDoRlt4Mft+9QwuL6dWAY5AwZfVMMNZ0mB9T2SQ+0fRI65tkg"
            b"bKXtsLJNOmhynNU0Oc5qmlyVROtnh0RPrraGZ63Iw4/iXjzqAHzjPn/PasYIE5dtqYW2gWFsEeKaeaEdADFBcsu9ElJQ7jylHGuX"
            b"hlvOmyCwU/9l8MH91myr/Lvp/qJneIIrNWb0BqQgiwenT3xYJPGzuLuJSpsozle8YIlY46YEx73prjhHFoDAKMs8BeVeKTNPsLYl"
            b"apM+RoJP/JK630d2pUqRvc3yBOgxFUp+AD/gBDP2qBRljTdy/Xiukim3jiyrNHpq0fkZY66Wo99BvU0xWj08aHA/UI7+n4Ah2BAw"
            b"jwlaB0dN3L4w0Cr1A7DFDqj2YOOXUCpOjBbGKokh5msrY1A8grAflaWvUvRnUZF7d/vzPWf7d1xtnbvdaTxJDJOHuuIZO9u9j0n8"
            b"tpxaD9ZN2VBkFctTwVLM9t134NRtvYsrOMT0lPHj5nHUVpnC2bZ4PXRcxn2oOn02qHo33c1pwHJJHW4sOebp8K8RGitJb0Z3+Mqk"
            b"NyQ6h63TyGtFrdcOi8Apl4HSIFJvU2odvydeBQRYYIiJjvr0H8EYwdYq93ckus+z6HiZ6Iac1MH878jvbkq2vBLCmurLxrdJURKj"
            b"PssctLHmvNsMf56VJkXcKteI3rZkBxLcYFzkqMTuOlu0NZq1u4R0qTmSoV3Yl+3Q2jbhlojGyfgh2eAdsd0M7Dl35c0zKG3GY7/d"
            b"PoMSemigt89eSfG2u/Ha0rh/dkb+fK5BUweWeRB5wBz41KwYDIlb9ZgDA7JOIaoIc8BipgQqg5eEBJ5eJtYjiwUIj2jS/zJJYOYv"
            b"F038RNXCm3yxoVATUzHHcQbUfLQXM6Bj6zn0Ve+NaVb6W1ONnNlgafW5JoOlCVjqweRn9Um3yU9R6tqYXM0kay0pmQldAr0vp/jc"
            b"Jv6Dlu8XAyB8OQBqpGeqIfyJR90HQLUTxWcMgH4lFtsBkEtlaKxKOQ1GEQzOpi153JN7hHSSo4lp2WCJuVEcDCdMx6+Y0gHHxKyQ"
            b"/ic84T5EohIjZT/lbII4a3BRHaL9lrvabRO6F8DUwQPvgHNO6U5DX/IV1+naxH0pw7Wj+eNPxfZIudeetCvq0w3Na+JL8bQFrWEu"
            b"s6a2t/+QI5yMZQlHD9UVaTswb57C+fQwmaif7qVt9XLjFcFLB/Usi/HoUFGZSvBaRXGXE+dlWNXLlB82Qb92cZ9LGHBUM6eZxkyh"
            b"VONSiSUoyjw3DvuAtIy1r1AWEOY4GUrYwB1B4LRkRLi/wnB+ktFuVsfv6oT1TVE2K2hva5jfElKYlMf4umhMgemmpIH1HdeFu2YX"
            b"zwrqXPOmK2CIdbIF88vIcYcs7ei+SW4acD5VsX04V6onRqzyZ2CVCX0rIL62M2aVWG39+ArSNG+msh0tkN3Vfo3oP7OoTcpYxnIi"
            b"gXtEMaJaWOYABalc3GhZZZSwgXDnJBiivNAGktyMivlWhn/WmOeeidgJY+JD2OLP+s9Y/jzLbvzUrSGY3yBvlbh051lchdlULXfm"
            b"ocPY4aHTvM+V52b+AOUAzsSp+SsHdP+MRFx62eQstisgRbt8xSAhGinHeSgj23XEn5ycMo4NdasRLBB0i31TWDF9fe6n9ZWK1267"
            b"c6oPdGn9SN3nY7kkDQye9RfZfScN9I9an/tfecVvl1d0wSaZXWa0Jy4QKxAQrrRxViqiCVHAwEKsu7mnyDsjAnMxMK4Z3jMkfmXM"
            b"/34Z87lHPXoDiAbZrQAIqwBdxxu3keRQRb+lXF5Eczkm7LzMkhm6XpXYor5Sulw+0jQglaZBy8tav/JcE2zG31rqfzWcrl/N8s/s"
            b"UgTrsfVS0fhXCC54MFqaoJH2QvhE3ELOBukJVkZqqyiPZbjwJqZdrDX/gdqNN6tnvmVQgjHpGs2DskmOSe9KNo/6Rsm6Grs+F3PM"
            b"ZywxgOmEAbrT/54LO96p4nlZSzBiaBRobN71uTrj9oHST75wkehuW7mO+GPDwpTcvryk26iPcgkpEAiuC2s1vYz0/iXwu9Ub09f2"
            b"Sdf5yiJ+c+055QqaZwfUAtdIi1bza/Rma3RyG0PiHYg71/06FXcrg8vvgeH+CjpO874XMUcpBZYRJAN3iAcwID0wxIlmnNFgnA8W"
            b"hLEx9yvCDHfYEo85Ud59KO/DnxBBeJCSh74KzOQI3hsz0nm5vGxl4JmK7aT5onab+am0emfWPsgmlDx6o5EgipyAkFWs6DVgJtIN"
            b"q9bO3FC4Ne7k1Xdistzne7Z1aPZtDmkpzHT+jUDbEcm9eBdur3s7cyrF3tthaegmgNQboerw+qZJuoW2i8NDGQYucNx/8acY42NQ"
            b"2DzZGL28WQB42/5eud6fz6pzNU31A3F8TBe7jnnz76v4T+63fKp0w2qtqFspq7ViqzlurBUvJR2+Z61QigWCA8Y6rgHUGUGDUJI5"
            b"q7jCTBuNWcwg1kngMc0Sjzg1cd/ARfK+eAvwXDbBX6A7uRKd/ETFyR7vfL7nyddImopVBJ7xOmTpdFAqKvbbGXnd8TgT/culY2qB"
            b"IY+WENoMg6seyqB3njNw8kYG1Wuqj1y69IFyIN1siLbSGmbNoRxD0n9gy9CrMCi9/fjdK4EZdNIsxpzmScgGeH4oDPlwVIkvgc44"
            b"Palkxtpn+BLojHegM6txzods5FmU4w0NsndPTgDIDRGynyEQeeFw8VoI8hPBzd54p6m2DAltOGbOcMljccdRXNuJDRwFkMh76bkr"
            b"DBAKjDKh4muBiH9PV+FIcdsWXbbN765RcQ1qSKtfTCHn2bbw86STg251wAHvyA1yMIj3U8/wDw24ZIos6QEvk5vDy+tkyvl9KgOR"
            b"zpJpcYJMCMGtHDnNUe9KKzyoKldyXVevrBrBV6/s/YOrR78iCjdFFLzXkihqHJaaxPosAKWeWIskWO+UBkJFTGwoVnWOaCQxUBe3"
            b"K7Gak0KukcFrWzLxWdq1d6QTyCsA8RseQHv2OlqsXYNz1AREC01AloNA7vCxg8k7jV5QO+4IOy5AzHQf8JEiN36Ii831EVIhlyKp"
            b"kLIbdOHOxSFFxIqfHKX5lpo60cKluGR7QpnBdDmHHX4KZxO1bmukgA8x13Ij6EnKmyvJnNXXWX/tT+ROkaikZJonlfUvb2i+WxHX"
            b"KRxU6ITvgR88aJ02MINayfBjDLVPtSkLiDHlYr1GQ/wfcgFZpb0DJ2WgwvMgAhNUG4qliOWbxYGAJxJLSk0s9v6wgvcHs9lNhaaD"
            b"l0pPHZgJ8mxauqz8Zy6HVPmtyvZR7SXaCvJ1m7UxM4xZsWtLTOlM1roumE9CU0gxlqkcffnCa6xEbbiGk2gxtwybEC0uir2pKcJT"
            b"mO5ja7H0FXoqxI1rGe1D1HWeTPeoCqlbRc2Bv6PpwVoI9nz9E4Rgvyd9fg3n4cuEYANhJkgfi0lCnAJAUgcASzlQbuLWOO6TNVIa"
            b"USESWJcAaE6cx7Ei1bE+/cesaO6l4EG2dd3IeyVdPe3kjRinyoK8rb/auQvsRe/o6lhLDeI9SXXty4KY2mS2MJm2BfENXBduZbji"
            b"mWQWZVBkX6JUxawQw3nJsaaxc5IvttxKOixWjqKUHtuaI7BX4aZ4uwjMCaFNb3Kia5tjMEedL82+xNxNwQwrlsZP97PwNe9sq1fZ"
            b"7pHQPLlknRUkLq9cE3BDF95aj3VPcuxPVgYJB4etalT+8e7k3oWshj1o+0Nc6b9WXcsZhHb7ipYG59i7/FKoV2AsxB2WYFJiHHO0"
            b"ChpoYDwYknS5ufNeY4yDsoYrwRwxwRrPveNIBkG/YuTfyNDsf42Ptvb4JRLg48a+s5n9dEY9MnmfEI1b/dY5tewmNqGr4/FRyfcf"
            b"4zbZQJ6K26NpzHatA0qhZLwiEXZcrdj3C+PdytebLoZKdljC88ZjZ6Qo59jNlqaTaDgo1tN3m0g0vOeOUHQanrGFZSNVIxupmoNG"
            b"1rQDWCVzU7UKKl/zrb9AyZ7Mq2cNS7h6ONCCR4bwMUX68dP3r7FN+LqErK1G3mBGrZPeIZocdj1LDmIMqEWxQnZEOVDMCIoltp4T"
            b"7VCIx2gV0H+hIXGviTD6Lu67fUpfmC7uS/XD5sbNRsmT7kaOoWw5rMKzHnQOTldKpcLNUbUpz+BJn+7Atir90a5HvurNRPdDrQka"
            b"/3KfeuQu2rawtBCDi6YtXGCtVrl3xMIeSNofq83wz3QqtDU6BGItQaDBCwEgpXI2fl0C8cpyFOK6roELoxgCxUnceREkhEl4p79M"
            b"rky+UMsWd3V03hU1u6VDdpe58VSubLTunXdJ76ma1cq3c+2xoc58qtqwaTDsqwvb61nUutvQlnUmtmKWxm/qeV9qjvIgOZ4ii1pF"
            b"I1L2tC1BkBIJv/xEx2HrNVQmCxXll/+vsViohG0vKA3HTA3X6fmr9Ry+SaPsRSui5SiUfe+ao7An5vhrLjzHiVLZlyo6BG+Iwkwa"
            b"7wxGymmCkLQIecodWBnLXR+oj/8CJNF5DQMS6+I0uJOM2H/aTPeeO8x7xfM07V0oSb7tJXGvOK4S6U3D28vBXl9bPqi9y/lKjXto"
            b"qLG1lkQ+Zym7QZwnrz0zZ4Y+m/QaAbK9h6wq9elR+fzlgHcldvAzaNd8aHeFgFgN7a5EyTulhbn+wpiNMyv41zn3Mythh5DwcfXO"
            b"xS2zFGlKCWeOBJDGKEQ9grhNi2WJ15xQbwLTPv5jYEgLsnbOfU+n7FPEIi/aECucfss0fcjymvdQB+ZUTjCS8irsaGi2mSJHSaiY"
            b"p9vNKqWr/AThyf5utH4OlU7QLJXdw5LEU22oM3bwhU/5ITWcNn3gHPsRYFcy6nnUBmDFo/YYabVNgDETQo6SR747H8MmXIOPVypn"
            b"sNP9a/5wqz7/ShHG75Va3NMWc1wroCYgSrHUGpy1zkltXKwVqWXIxzLTKcmVA2I4p1oHBt4AUEfUN3nO3gEZoFdutGsr2o53VKj8"
            b"BNqZz4DEbAVtD4x9G4fy8sfZoLu6x19usOvPNG9v7gT8QbVmblvL+6ZnKr6wwhX1v2xpaBu7sNRtirl0moyRpUQ0nY4R0/a6JZJP"
            b"klnGO8/oqfFramU+LNlYlcJO+Gme5FfihqTSzsoT+jOHHU+uiraRj9k9vDEp+hmernV9VoE7vta5dUxhTiEjiCVx54sdiomLEG0M"
            b"F5bHpBUIVsBJ3vrGCkyC5SZgT+I3lQAFj/5ZXcM7uiWPVAUvJE7anMeyViJGO05d7MF0QL/noG03Sikf9bDa8IXczOEciJ8LO96e"
            b"3V/L14xIrJuaNM3dyLegRN9kxT+QL9zUCAuBnkCtkjW/NSUs3rk31QsfuS6s5AbxUrxwd5zFleDhCY5dnW2nPtWKWbuwyi0VreqI"
            b"X+HC7xYudChYK7gXlErNk16hUh5R6QArL7zzIChDkkqjZFA2AOcWBa61Bcy01t/FIeUfk2W51mR535ubdwCnzpub743SCThg8OaW"
            b"naDIxdozl1WZeXPLSgaLtxvbFJfdDhA542uHbnKH3DroIE5VG1dnRFtgXIOPpWq/jEYKJgc0O/CneKinNIEVGWDl0X1NBpirgDdY"
            b"0+phxw74USCnunG4GrXX8mW3wEyfySl1mMUdIjcmEE0FAaVpLIeYt8iwII3ToD1TWnEcDPPBCqStTGaJIIVHiP91BeyDsvN9daiJ"
            b"UOEzoevb7to39cP/gIhUfwmPJuiYyIPuWspdXkFOV8LgOfiBxjfdFcFX6lGLa/ss9ajHbjYPcvIayoQvZaPwpWwUvpSNatJyJQPV"
            b"eIs1/76K/+Sc/lvtvlPtYkeJB2NiUnAIkPQ4eG8k0oRJA0YphjSnLjhiky4BMsY5IlkwGIFG5luZYE2zFV6rPcnHTcFJSi6L1E3z"
            b"x46pNF+Hpuyvm6AmUg3W0TVNqgCUUhimCFRDlDoASqSFJsHW2kMCyu+pihazjvK55uFjHo92Z/OZWtXQ5U1RcwxBPmVeVQlpAWR8"
            b"/waMKwQpjj5MHfiA4wOo2YysBO7GEUR2x5zShB2GC7bwmOtRJWSFh356CaDVqvBQ0OqZXzneBaqqFjKvinO+EcAqHdqKKXbiZOvH"
            b"+EKDtraMbG0kx1VhxSnbDSbPVsiP17vKp3u1Djzgml0rZH3LOsB8/D4hHII3BgRnPDilpAQfMFhHgQThrXDIxh0BAmo0U95CXDWc"
            b"8fDDZWfqnfZnNclLITj6EdwRfhlEBY4WAetA/hMLiHg75dH2ngtkyaJIuNfgYuZ/sKP/FRlazbARBtq3zhrh8WSILXrTLUWM5UiM"
            b"0Hmx04+UP3QOi6sqqZapZtnLr7V6M9dKMw/Iuo+6ILjpAfePP/WVF4/+AnzVVv3OdAFf46g+T1ImZjfJRAiKe40JlhTFmsUSAIe4"
            b"IYxbJmKZix0EAE2CJtiD9igAEYFwLX6ge8IMFTUV97vrsSiqrTs0OIKFC/g9cwO5FcHxZ/njbsr7TUL3RI3mqPSTT1UQOrEUxc/i"
            b"mk91EwbFmRy4FZWy83xcUqVWjZohjV81NMbzlmvYGxrD5czbGWyLjT/5tvzsOLD6sM7Gt/xWyzHJL2z74uxbjKL1gB/7UpRTpYCV"
            b"nULfcz7+2J+IY9f9BzaIWL+Qzq6FEZpXr62AV9iJgxcmaxNfWRe3M6Xs3NU9osqf+q+dwnfbKTjssaXEWxm/rZIL4kUqeIXhXKAQ"
            b"a10Xq1znlFSahXiVQgkXFFAMzgWN/0002modeQgcu4PgGilZh8QiuWWd1rcsRuxqTmm0ImCpTsM2RVICqkJj0L0/oB4C3NJpEsDt"
            b"JwDS8s16BkebIWqvuAUrRK1cImobMNr84UtE7S8U7ZzkxS8vl5gB1xi8YT5+mxGznClEiFdApIl1q4AgCUhkffwHqqWL/0CQcgHe"
            b"kfjfuovzrfqHbBi/d8NeVaw1y+qpn+7gGiCHkvvuYG8C9/84qeu+La04mg4VbvhsYvYMpwkuGHZA71w5+4W95GQimhaHzWWSSTkx"
            b"NJi+jxD7AYcmY2sbNnyW9BkOXcZ32FqP0b+kVpUlrcTsqu+6EWVxTZTFFVG2MhHrnowStBPmbPWwqWH5X22L/iHzxvetCGh+TKe2"
            b"BI4oiRgxBChiAWErPGaBBY8sU4YIL5njEFO2MsIRHIT0kif9b0MsxZh/LnXri7wGPtGLfEGDEldaBXhhBPC+VgGuashRq6Cw8Wlp"
            b"QNzZuK8GfN09mvRVFlKxT8UK8oXGagFq4MdKuTbHbfIGp3Itf6xd+6B85QSzZ3OulTQBvpAmwJfSBPhCmqAZZVXCtdXDKuDUsH0t"
            b"TfBXghv+EIRhz6gaECLGoIDBWJNgDOAp5UKEIDxxPP5f3Otj55HGUmKeGr6KU+cs5w79E+7ib2skDgT/QSMRzwyznviUtxqJ2wy9"
            b"E569jWtbaCSOjLTP0EjcrnXA6o0aiUfBPr1btzQSh99E1kgUCdiLj4MwenV1OXqXVnyvnv0VR/x1JX+sjBjTcAguiLgZos4YkqZp"
            b"QXseDBdEMs+TLIxQQRkVy1kLiARwjAHVgrIY+KfS8EtlghfDuDcEEm8LzL4rFDDXBpuTDyYV4ou9en/acsKecjybE97RZdj3/Kem"
            b"C65kXXrx2ZUCAhYHwKvHIki1a7oAY7DPvmT1PniqCJajxZVi7fy4hWLtU42ENDl9mpFZRRBmdb7t5LlOzQNZGYfLxkR8y76yTsWy"
            b"ysStJkKXfHGGlv1L+gkfl0n8ukTsGSGKaRQCloiI+N8gqXBK4fhESGWDBUsd5NmWRNgE5Vl8agTiyn0NjPcj1A/8ossLj6gfU+DR"
            b"XWTEwu8AvaWveM/9dcIlwRXiauyc7rX0Nd12NI640PBKnzhdaU8VmbJKRqqI2NJriYUPckWm7uH7LWmbNP33YeYeXiFE3uF/nIdf"
            b"8z9ysnhehD82Dj+8HaonG/rhAP+2z+Rl9S4vq3d5sJkPdEPfXl6BIZp/f2Ib/k3qja9gv8/pH2vYb/lyTBQdvxP2S7znwohgNLVB"
            b"U0o9TXQPRQ3TThuhqSSCESsII0qCFUEjG4v8uNagIPBfYRmOLmEScIWUGEaIZSmQewWZJqNdYZxeJS3d4p6x99AEvzL2HtaVYutd"
            b"44fRXt3CbJZXVqKJD/jMi0NtPuAS4MDy4tmJB8uhfRFvpRIhhQxW4R0be1ys0PALKXbiHL2LmUiZmj/zCpe1VXhLiW6yb+MTLmtj"
            b"cDl1Cce1TyWeuYQPTAp+Uia+3IXn4wCK1iMcTjWVOxCKz/MIdwkyAcoZr5VkhvKYybh2QSsrMPI6cAoOUWEotjEJKqK1Zo6GQHT4"
            b"BO+zz4SAvXQ9u2DHXYK7xunODfyS3NqvvJf66kUYY4SQzZS+7wikl1MeYayJgupDbe/JColB1U1fNVOwTREYId4rpJFhrHZTdg3x"
            b"XXZNoIl3T++ZlsPiD9zIRHRmZenlvD4c6ewZVSz9lp4ZP84QYLs42Y7matFgcwQYXiLAZjzfls1QnHN/OuVrjf/6QvrWMPeiVhib"
            b"6rRYpBnOSOCOOBbTFokZTWqjucPYxm+PjTtgwoJRhiklHRGBCmV/FP7rakr2SSOyAa01yMZOQGGsA0mlUmmOeFrTwa42voVGVRs9"
            b"VnIETeUFuZc1pssp2eyCz8VfCk/M53hq56WR4zfQiot1OTvHMdSm7T4QcAl5j6xFn0kukopTRSrwFKnYVvXjq1dWbC155DJ5qHvJ"
            b"Q9Pr56lhr8b/73O1Prc+c1gIarELzBnEHRUICceIR4qx4BjnViIIzGhEAwIHHockUMAMcOzEL8T1EuL6BHP6iZCu29iBdrrTj9nn"
            b"Lb4bkNzb2IBS0aXVnR4CM0d7kQ8wMFGJS6Iaerori0/zag7N74A5VoPQeKNlM/m8ea6UjsOc8uPoyoGs4/XlqHdhWc9Uuuf59uqV"
            b"Vb69emXPt3ceserRL6z1U2GtDHgIwYM2KGiDsTKBEac4BxKzNkjkjJAWe2AuVq1cCAmIYiaFtpwJ+zt0+suHTivH8rPs7RuJIrdQ"
            b"Kwhs+esb2o3pnF82SxppuvBcd2ycIN3RHfuZc6entmD4YuyEL8dO+HLshC/HTrWI7jl2uiUw0/z7E9mx37HT14ydGFdWWwqUYA0Q"
            b"LAJmZepYcKu1p0oHoT0orZjhmiFhrUIuSMYgyJhq/zpByhuV9B9TefxACT9duZ5Jvj9J2ykoT4Uo5U1enCbgh5V8+gUey9/Wbzl6"
            b"HWx2BC1CYyn4gR5lufpywF1lye+WVX/MxbgSomRFdez0HeOsmpmxU5WyefJaq4zt86/u4UqCcgsp/34R/ytC+e0ilMxYDVQJERLi"
            b"IH7TUfAKOMVOKim1tipuGqhxxXVDa2adlKlRxKmRBP7F5eDzs+1dZtmTDtFNUttMJRKvHSE35crSVNm3QIe5Lp4RJx64h9AtKn7j"
            b"tzXnuCFrE5F8HfkIzDBtHdw3KMTHvETS/c5xhLCK+Hzc/pUHSLml+SDADLEOzTEZRebrL7FvLg+cfUoP/qo7T6rOz71X5uISZz++"
            b"7czPHv1d3fp/bhEAEjBWlGLCbNCpZ+lAgxEJ1WwQRcoZ5dL/AmHUYma0TpMCLqg2Uolvcnr/IMuvdJguekufYBS31853jeKu1N1q"
            b"zm+3TqyM4sbJ6NRTXtw1iuvuysoG+Y5RXNn8iA/q8jzsrqRF/KnV5cwnDpY+cXDhEwdLn7iGllEx5Z6R5n5Gh2TGmfvybscAzgDq"
            b"ldEhcIE5N96JEECjuAdFoH1ukgtruNbUaKWQDIR7ziSLmYx48mJy+esctHYOSteQ/og521UXDwXbvh5LIa2Abf7WNWHp9ZQXFK9M"
            b"2PCRAzvPohT1wDEIlbMhvNHxqkMG46B79kLpVDngGyyDnnterszIV6ZB8kLXQV7oOvTcgneJBr/mQb15EDBvnVLWMEMZUT6WYNZ5"
            b"5GMVFoyOZVvcTAXNAneSBidF4BBft9pKoczawPe/4H55s7P6YBM/lWpYMbjwaRaxV3xr7YRHO/RR4eHIeDDoPMhdoxHvzc5TEoLM"
            b"G6uJDZwgEhSTVjx8bvRGtm73hr8Qx36+70PkT3clF8wHD4pUMNOC1xO8MdGYNiLSRZRQoITx867LvcHQ9rTTb6ZEvmuB+TQnr9XG"
            b"rshkV2pja8cJeejg9I8W/sT9tnzTzvndmP/JjTkwh4wE7RDSkhkmqIe48yZOorgvNyLuxAnhwlDJEbUeGZCEBEE4snG9+NACAMcs"
            b"5QMVq3hVtPYAhHcy+zQxLdaSvoYbZ3/d5z4nWnmRoERS0lXlp9rysVTIbbWgpCllzyytOkX1e93cUVOC1L3k2Yq1DfnIZlm0ey+T"
            b"oaDO5yyRGBiSjVvzed3N+dUm45jWMMpGoPL0sD04/Ww05K4Oyo3pfFmYxz+AvQvMD3QGLA4s4eW9PjTDe1qxPxQFJhuN47Q7buv1"
            b"1r5zz/CkIn7UlTypxCRIIzJR6VNOm7WzFu25PvRrys+p5ldS6nW6X0mpb1X+DSn1l/X/Ny0L0jBnLNPEJAFizLmN22vhXFwYDEbS"
            b"UE5NXDI0UVQLzJyiznFGjdN4jeH4e3l05QtYfRPrT7Elqw3Wd42Au6d9nmm5OQ9TipvCdLdjWbJTTmrwNS9lr2uWJJNpCQyZIhrX"
            b"T3xyi8sfQCPKk646hTDERGVntJHyWu0klvc6DPHKam0I4tlm7U3q3L9nAvRLm6t6GGC5JtobKRRVDiwJXFkvseWeY2ss4UwzFJyn"
            b"HAIYZqli1mFliMaU/ou56o7tw8ICQbT8ghggYT8T7HFymPLc4RkvTCTEHRMJMk7QPsdEop9AwWYiMbo+oMes6dH14SHl99f14T+Q"
            b"voy2OCDEAcUtiIspitm4pQiIIW1jjkLGEMsoUKTTjjOEuOGywF1MetKD+FdbsA+hpjfxQTkk7Vq3lJHrkiUwCFUEMMpRs1/9GCIo"
            b"x6T7RAhDC8fFJj6FldsAmCoyJez2oKN0ySX6Jhb1DyCZN+hr/ClO3R1Zb9dbRzI4HMm+AyuLn0gurHbE5ZVqsNXIJ8xhUlev7Dvp"
            b"V4922FP/qKVK/PZcv7/n6pDFyGOlFNOgUMxxIJDAxoJNvmREIZ9IERynYtUEjBwDDVJ7FQtd+hVQggeWvB/1trylqHuUk5f+k516"
            b"+Gr+1ae1qb9jPknufpZx2pU/5gKmNH7Y28LAdd0rT3/yvYk6PXflFk8qUMjs9CWGs0aG99Dg5XNR3dyMliDOu3HVCs7E5SLDS4Bs"
            b"7yF3NY3jc/R3PZ2/HPAhNd18j59VyvN6eA1qYFsa31uhuNPirfUT61ZoPyCbt0c/te35qRq6TfpdgBg+UUX3c+AN+fq23HtkXU0A"
            b"mVhGi1g9U6WsDjSZ9GiFLI11tiIifo0w9cAMD9bEPxnKnAkeOaX0X6GGeN094FcNhN7iexDoY+Msrdgm8O2TqkXYUhSxywXXoojD"
            b"iOdaFHHmQ74QRRyr7fSRiiiiAHkOgopAHuH5G92wDgQUU3aGMKukKs74zs4cs8oEYhE2U1vkvf8RzmqLTXthT/53U2d2HH7UZljq"
            b"JMoLnUR5oZMoL3QSO7jXoEPLTwDBl+sk3mg57EVrNSwqLQdcc3dnSg9bGi7dibH18OlSPOCCc9wGZZ2Vscw0AjNtafwhA4OAgXtk"
            b"pVFJJtFai53yJm7JhQoUg0RfI/DwKvF9pHl6mf7EY0vXWW03zWv3GpVjwiUz0bNZ1bc1omfvPUiHdzd4NMCpu67phNOrGJLTVifz"
            b"vhE9xxx0d7J8lmzcQ/e2zNGSPsQUxIxHluJx/M7iobW7Oq5E52qVUozrX8eEz5tD8q/tKE8f9nAfQ7iuILIr+oC8dNaRFxSCLaNO"
            b"xvGc9Q/J/0j3r39jh/dGGv763u+sKxBAhJhzQUjHBbcx/2pvuaCUBWODQZzw4C2WMfFIxMDLQGIR6xFSnmH16yM5e8NLnQDxngbC"
            b"orV5S6GB9d3UvOuncmioTj5LCkurxOCN08KEOzBXpkrNwbmD3BktkVSV3up22YeZ2fSoHFxczdRx1BQPmwO21es9vbJSqj4zzcn5"
            b"kzfZlLNThIyfqgTtswIZLrIBp5nk+RS2Opg3Br68Mt45fHOOHX/zsDPWKVJmRajg10/yU/0kIfj4bkHEzBpoYMhorbCxOF4Fixt9"
            b"7GJdGwI38QuqlSQCxdTrA+FOgIt/dO9nVvg+PtZNMOpSlWXOaFqrskwZUyWszOKGgVnhYpGPjptgJJN9YkIul52Xn5sQ1RKW8gIw"
            b"0c/YZsS5HJbXLuAI9UM53vvu5KCYLwWdjbvagWnx6WkbqU8xpfA5I68N0ll7yhyDrXkNi5cU2Csb3ol9wm5Nhl9XqX8lhPTPAUW3"
            b"fMohKbdwIJYIB5Jpy5STTkMsWr02WFtFgvHMcGA0ZlXKKVEOeYGtDOzv4wx8LtD/Vg2802P5cKbVKOwCLiGGzmIHl6jnOk2GnkEa"
            b"8AHqaGMvCGmjasBNwd4VTAJvUo6pEUuewCTGRSjfhRIcy0tKGsfgA1tBJmtd+phZIpgijlcmwMNH398m8cgqiZrTBximdOH8HuWY"
            b"D3EG5HPJ3xbFcMr3zhANsHEEcMUyaGWCZ2c7RX139AIcjID633rJ3/bYcsQvX+BP8AU46PgFDTIWzSQ5U4KgntMAwTisUCAyVt/e"
            b"aMdisa2CVJoFyxgWFoy0Uv0gDG4NPp+3kdcjtMcY3BE1O2Lf6Z6DkSSnSa3as0SvBCNKIEFKnW++HaGG8AUkt2khpzNlBStCoT3l"
            b"dpPVCFe70T6frAF4AIjl98xG6wiJoyODZ++ZI0q/5/Th4ZOwwYfnma0YJvKp4ssMlAtLIx5YgnJhCcpd6b2c2fE7jHg+7iO2huS+"
            b"dBH7REgu54hqExwDh3D8YVkAjUVMZUJ56rEgASmhvWMIpMZEx0o3VgPKCyH5umFwBdCCiyS3Hpd9uxj6OU/KNe6IuGSlWma4bG3Z"
            b"OZrqi8Va/moqfjWHbY2mhL2nl1yIZBV9rJwaecVJ2sc+U0WthAHA5JAEXAvS5iicjR+26KZmnx0QA/NtEqQcst2pziKCDRV53olI"
            b"Arz9TYqxIifFtk1uLhTXSuj5jCn02D3gqhgdryRftipqDlvD4aF41vN27i5Ge/YLzhJS/o913Yc9Za4katM3jrO6PVw/gwO9dbZl"
            b"u75tJUP7rIn7zcJaO2ew6tWupbVezMdeiW59GX6LCxYYKI4Vi0mXEIti6cg0ZT5+jURw2HPilAwg4mYOGEKBA4qrvQYwQqh/x5Di"
            b"Axo2/eD9Qxo2c7DDI9nvx64UJa7Iee9yALXEC9qTWzPYy6Gbv/jP83i4Z+++qJdHTgXOQbeN3Vfa41OgyMineJjyOXvUXZ4LFlwR"
            b"KlhFlWATQsXYfrigUfD+4QtKxa/lxB+ynOAiFuo60PitjKsD1oBITFnAgJG4CDAvNQShtQJsUZCaIxW8oaCDwByk+A9rjLMtuaef"
            b"ctQGo2/qcN+kia30Y1YCiwzt+jE3+HjxXUvKlKdy2IkRHghr5ZTpZ2UIujp1jkpnTorbB1+kPftKoFttN5siKBxj2Lu//DikvfGP"
            b"tMazJtAmJLQV/rw2uWhPnU9ZIikgJHsuYn8p5aJL6PcaUMwNidguG3kkfN7sAubSZV9mQNF4t/8aUPwIuh2PNWNwXAZHBYk7A0Wt"
            b"jHk/bheYxYwYGgyWiFISbFxDlFbWMcSt4yaWcQp9l3Iv6zF4PdHummU37z23nefOPfIWEy/7vBVaxkteXK/6ywaYcaP6ewCMx3Hm"
            b"Lcpe39/gVX+DDzjA3NpoVH+nZT/LQWqXkGygxGSY5aW43Xl+l4Hc+/K99pkqkY1G70OaW7olXy7QOxUUl8vB3kyC9wuEef8cp+17"
            b"mWt70vJEY6IkUpZpgQUiimkarNGBYAImOIKBGueDAQkxqwExLqY2742niPx1Fe0jr5tbteXn1ciTLgXZr23apfha77NcLMopanjt"
            b"5XPbTvQmMmPeZZiH5o9VSBQDpm7asClh5W5zxgmbSFcsPymV2zGcCX60X3iFsOiub+uCcPJuQUvJ54AqVmKLWye6csa5B7fAcyjF"
            b"dEjIuh5277j8W7d+e90qKLIUOHKAuUGa8sTJI0EIbpHHQmtJlLUs/gMOsWjlVmnhNTBLtOWGfytBr0FIwGuunfwMut1U7+uBW/Pd"
            b"jmoKiYES1U44K79edKwVipL+ALhBDjwFauVNufdeO3Lm7zOqsQ3UGJz/VtUAkqwMMUtz4fxlQ3XIRCA534IcDxQr3tyN87gBVJeX"
            b"T4qg9jHCh04T6ZZO2E4ekzfCNUCmQjW2LewYRisY3ZeKSa7lfVbCkKR6Rrqoy7MtH7WSQLNHdU/7p3P6yt/Hq0RfVoOX3euvZPs9"
            b"SPRaO2IC4hQFIJrSVNsz762PF5UQ1owzRClFOkjr4qs8kQhiVUUtceabwHMnPg6/gsixK5ScuJkAe9+vgR+NKkZaV31vEIm91qyS"
            b"R4thyCGlMsVETpram5wCaSFzWfVCpus4QHPLi8nnTV0CvM0rDzGHyiHjJS1mAM+V06WfsiVMX545X/Yorrm8+A3Mg5ToTOpIuzKL"
            b"7Upi4c5g13GT0ytqy3xWuIOxsMHQyW0OPJsUku7JycR+BtwrRcETL41Va/lw06g70i910detlblnRm9bWWlhVHTtn47oa9N1+XNa"
            b"p+svx/pdElsEAywx51o4GrQ1SGpNeMzJjiX5TS2slRwZo5mnChmLMfeUsIQVY8HCN7lUPgOXXCNL3oCVzNEK16l6RVC+g1EZ2H31"
            b"3bpi9+Ep3iFNBSXtURl0+BAp6PApFvgADNYgvdGneFDFOI2TxDsk99FdCV9QsafWSnNL5fyZcty76L3H7hNrv4g5mOM9QeNKXq16"
            b"eC/D/lvAjT8Ez9hzqSAaG5201JBOW61YaxnJPAONJefKCeyxVoCMQN77wIA5haVENBgv/Fc0sO/Y+17TB9FLuctOojGbLuxWCTUv"
            b"rPX5visTCVlloHZxqHhq9PlwL7s5bH0FWXlIlH6JGITOSlTxiRCnjlqZt8hZuOBbsxqd9uUzXMTmnSFapaDepx2VaRxFBO1Vr9rz"
            b"IesRKFsX5DCAn44ZxXY2HKtDcYpenGdtLiDF7GpATDXNhtFDucRQ9jFly/zduN9xkINvZsu8nqVTvKsHyVo8SFZxMzIKPpSD6m7C"
            b"HDlRQeh+3CxwQdP7GnXLL1MPEsoEFbADwzhGUiGuOKLSQixasRdWKYk4FdrGLEydTe4+FjxYYwwVVn8BZWXBV/mWWrZ3Jx7qxUIi"
            b"3v7ct5puCpIgGbsjD2WbM3Cs6m4RVmQpViXQ+pSrki6Fpc+pJK7DYYmQu5P1KdmrypKTrzB36Y1zHENyG43t8OB2kirzLApJsWl3"
            b"jgHpuDdt1vPnetJ2nQMf8BL4gJcivrsf2jliq8HGbKJF8YMBwY+IIN9I99iTmA6aOCdsXJI9RwA4KBGs9smmMmCNLWOKBGkIUc7H"
            b"R5x4FLwAQhmY8J+jeHwBf+N7GBk39v03p1ijX2XTvu3ed3OgzAUvUMFPaEYjbd4flTt85GiaVty++UXN/CfRnrhX9wRvl5R+inr0"
            b"dXVpJZhlZAVW1SCzvsYVxiIesLlWMsEbt6OyS+msN/hhvfEW9SN//Z4AJa4guqc6G2vE2fZXed2UbZ7AshPRwCMq0eGG8lfRrOeM"
            b"66Pqhc8Qyfylg7w1UEtLRuDax+SUHDMsM8hyjhRG2qIkRoE9Y0ZqiuP/OE2lsEeKCU1AYvnvrCMPEv89FtoHgBrqg9S/24vdomFM"
            b"bvaMyYO2MTmTessmeQgA3KXvSWHNv2BqlzuWg38inzHdp4NV+NZSEVcmRhPZ5/5iUVI8Z9VEjjeLAW9Wg0rmU25PTyn6WuRzf1Yx"
            b"xI9nTd5vuOBXHPHZv49Lye+i8d2LhkRYOI+9tIRyrSkQLeIuBBPuJSExUyFMAVzcdIBzkkmFvAlSYWKlt96+6KBMSSKf5MhUyRot"
            b"+9r1zv6zSIh7ha72tiYZavPc1d1lMU+qYGd9RHIBv30KtXsGsd43I4VQdPgRrQJbY9MD4QAD/mBh4zR+Ul6iyNnCPtHfbD9zf3dQ"
            b"6f7XRsn1QXyQGRQHv1Iw1PbK9+he64ltWqVU7ORNNbsj6ablkJp78qWacSNI7UoJiVRxpDlmReeuVeFmj34aaHnZeNnayR0V5RYI"
            b"eXf3QFXGw5U/yMolpO4oH24gUjsfKI7vlABmsV7mIa78ikjrbLDB6OCBi+RQyilDnmIekHHBccM9d+63ZP6RJfO38VDqW8C/hoby"
            b"UbGLz6ahHLf+CQ2luryuwTJhoTyVU3pIDpzTUOSShiKXNJSX5MATeHEBzDjRGKcZ3nn0x9Fuv2XxO2WxQsQpElRcC1BcGJRRVshY"
            b"CSPuqcaeGkbBahSoISJ+SYF677n0yhiGLcM/zibqpb4nXIGXu7/yKSiZ3Eh7M+rK2lBq1Lybui8dV3574RnxZZu+5vY5+jPecpN6"
            b"ZVHVf5oL86n53bznKlUMotIlVALNLTRwejk5uPhDbcJ8k7vSpvoURxslvIfQZKBcCqQeJXBK6i5H+0xedjVOuz3e+O11r7L6xf1J"
            b"JvRMzPjqPF779TVPPqG38adESEdVvJLcyXfKk87Sc1YnldSjWJU7EjwCQ7U3XBifXKNYkuGn2cXUgggMK2cUI14IKwH+csjy215R"
            b"4q451cC7qFsLB7uhx4Gcrk51XSwWSJUbFIsJpaVWOR0pLSl0F2m6UBPNQRv/pcIMV+Vwpz4qt8D4Ux4QPLHfir0SHukheTcleSX2"
            b"ejpYtdEpKN06QIi2byAWb5AjywW9i2V+Sg05nPhqV76jEJ6BRbaCm9cVN79Rclcpdo5s7mrpmibyC2j+TECz4h57y7GxqTlitcBx"
            b"wQ5xV8nBSYwREMs0I4ZpzYJXSCAG0gdQAoVYEL/jH53do1c94g8hoP9Qp5j/b9krvtsZWEKle75Yiyje8MR30dn3QdeohxOne0Mr"
            b"0+g9UbeRyWuPDd3sJlEPHdxcqVOhJg3wo6fYdIuO5g+QO6olOS6vlIRJNHinTHskBUGewicQ78lV5evfkdxvmQCmzcITEMlKH2Pd"
            b"Wn7RdD6gI6SHjqzVNmpkSae28Rcra2zN65lM9GsFjfetrGl+TOe21koQIY2I5QAoZAjCShOKHLYcjGHGiFj2KmSDk5YpiQQkw1Vs"
            b"0yzPsBdA6N803diabNJyA2T6LeGleRZ/pHwpDnnSSsFzrk8661FUsqej0lx2qM4t8mM018Ac+9HAwriqnubhrveOSjBmwJr3gLmQ"
            b"aIrKdGigm2znJq0xlUJleAtsmCtzEPmUt/KFQL+8qT6Hhs0z2KB7nFVW1geyg5KdurL1QKpnUAywT+X/6skFvq/+96rZMQdy/Obu"
            b"T87dTgstZPzT1Cy9O5PEW4a9MYppSziXxNhYXyMlmeJMxSNAW04CdxqZd3J32aB/QQtarFvQ+KoH8mUCSg2gek5YG5K7GByrRn8X"
            b"tLB4ySEy/eeAJpz6yPDS1WS7D6OrST5h6ScIPgosdeZ+99SV8vumOyzT8PcUN1IH5rrrm+QZcembsPOSWzmkrq+RoN05OKbVThlq"
            b"roUktsg+Y08mt0PGfqhp9JBfw0qL+MDa4Y5Rs3WSmyebHHOtxlzxaE6ODmv4OnsxXYsYLWxfD2UM/NIJ8KcLGV3Yr7wWLPqq3Kwo"
            b"dVp5o2OBHQhoE5SVCjT3zmHqBYeYoC1mSiPqABlqpVaEgQ2Bcf83qdHdlJobalN8fBGG2vROfh5KxjoHTUrGB0CVmS1qDf0djQHv"
            b"jfomjgBnMdP57q2k/uVsKWK7JWv8yUcGzeSCr8wBtiuiLy0C4JBTpSOsOQaVywGCi3Xr6REwu6AtLv2E/mq62BJTrvtdXbqnPeyt"
            b"WD5ydf3sWvwo98PYWa7fkz+agz52DvkI9WghIwetRn4GreZXp+4tKIiOeQTFtB5U8JhjrT3FECQKOKZ7sCIuBZ6imPidkjQwk6aS"
            b"EJcFyyWz7r9cnr+svW8pIf2W5//N8jzf/CfCdfPyXF6U53JZnstleT5A+qbJnQ/Cdb/l+ReU55ZDEEYIF1MwJY4KrIM2TloRAhUi"
            b"JmNpvGExwnsOiidRO2msxs5azb+JvbL5oS5VmfBeYXcpmVR+GmOzu9MrHprd5XeuejrJIRECM7jw6WWCUXvhy9w1lWwayX3HQA3m"
            b"EOkSSQlGG0kPbaoj8wOSo22adMqd06e2uWm3W8gRBwx8z7NNEN9GooJXt1bsd6910WY5TMoTZEKWFxg/QQqMS9O+SzhGi80ssUQ0"
            b"Binpr15+mVrS2hNww3RUbQhSAzeqEpc37QpZNZxrR9q6PXHqJn2yLpL8v0EC4f9qVbPy/NBFwr0uEtS6SLUfbOOR0naOiwLS2Tku"
            b"JYfsc2DGcX46a0VDUEIZjYyKOygtCPaKGkyExJopCkh6A5I6DiR+WzlwQin2AkMsUkCyPyUyh1/5p7wa3KnHmj+3J24dpJnvVztS"
            b"S2oMMqmF45/NAhcYZFKJyotrkcvTF2Qm8v9o0HfTW+aptWCBIceftCGmv5g40k2gEziCpj8xf5sclj0J35vbcRH/KuQjh9YrDDK+"
            b"xCDjSwwyvsAg43os17SAaw2mc4zXPflbJndLa+6PzO2+CISsBbaEgnImKBWQp4zSgJkX1EhnrPFAAWMLzjDKUjuYCqmdlbEWjTWr"
            b"/xftVz+AZ5sqPd23x1rRI7oLnOhjLrWMCucQA3+dQAeo9ZUp1TPtpgrPRrZ1bgOqzKXryhVn0iQh6Lx0tobAlch09al64DMURn9Z"
            b"6UaXYECYQE0zUSvMXA7MnxpicUwnsqPdRylh5YO/aVL1WbYlq1fWtiVXhiYLO+0XFiXPbEt+Daq+qB+slfTgXaCcCY8E9g45p5EO"
            b"znIdUNBSB7Cx0gbDdAhYKkCAbUiyo0zj7zJW5X3HoWkqsFd9heumwoXoKGkbCmmfnseFSG4b+52LmDdeHd0jRxGk+LG9PtrGXXAO"
            b"ystNLBaHsRcenEihxNEtWcFRPLPFAW1Kl7OPXjFexOmCCi8XTjWzjeKF/S3pMdE8LmtCsk5hSmx3VO6dhdWHz586hX+kyZBux2cQ"
            b"rdcurGtJ/FUybfxW+4e3uq4/pr9QpcMNyviku/CJcqTaehkLr0R4ZiamLomsTbLJQuCY7DxGLJa3UjNBiNfaeWJicpMagdGaev3j"
            b"qM/k1WDr1VTrOad5VPEkHx59tZ7WF1bVh3fcaBzX5rSDCzx+zDUpe8xaPd1iV4hI7y/GtkRhWxxlJj5aw2pKzeAlxxKmSD2wY+Ne"
            b"IYdkaDAlvDUt7JWgsyt1jhoV7dHgnffhQdZjaLDcBe0PZvIpvrY9YbXUffVak2hxA/wtyOBK656SSgxuBQrmpzNqJfDZQ4V/+ljr"
            b"Hrn54VDri/oKhkkw1CCHMAQW4puDoywWnc7r+BX0WgINMesyJR11QnqltRFSx8fKrh1Q/+a+wt0G5SCCOdXEmYtgzl2c/1xD40tV"
            b"jEZxT6gLSTrc2mIMRXO9i7E6i8iKKzd/q8FxlVSBo+Nq0dTe9J5kNztkS4/WEnwKg768rPIxygFvNhdiKSTTOPAZhHiu11nsnDir"
            b"sGenzPMeeAIQeC3Uz9l5nvbZypmk6hlXBI8mpPz7Rfxv6+HbWw9GUEosdknEGTkqFXaSE5PEnqVOnD6PjdMhVuIUB6vjX5HVTMeC"
            b"HjEXkH8Od4AlDO2yVP+BALQF9SOxkeE0VZr6GbVynifOtyH4wibnKY4qtUbfroU6yW6E8kqok1al+muhTnLihe8LddYHjUKdk6Vw"
            b"uiUQR4O7kufc/swu5TmfwsO+Xp/zHRI1m+tzNorHP6ZaFtsXeRToLHXxTKDzVV38qVAHo5SHuMCHwFHwSsr4bhC3hcbGlZsKDoZo"
            b"zWPq41ZZhF38flosvQcdgkfir6uFb9aRbKsk0085jqdoX3OWXd1BKmhZvH20KuU45qoCyZbvA2qD+V7LUnL2LasZU3/qdMIMao05"
            b"gjU7dDJcSDllxpERxUc2cfm20jdme+V8JZMBQrI/dZI1a64kB5VeM0e0Sn2NcEXniVpC48/dqvpMr3h6yAnZ2Ps4pOrktMV+Cik3"
            b"kSgGdAf0bR+8yuPtDcqx5ZJiyYtwb+U33qsSFm/qmyXyQ7vU9wymVwjetZHqqRmUBT0XXIvu4dECGRHAv7Xw99fC2khGNWdeBm0T"
            b"eRqodAZTi5nnjAYVtGbOGRL/H3AncCyEDbUGGS3lP67Q+cnSm3cFP/+UQudT7c17Upr3yIBrZF6BT0yr76zSif4nRHUJVxy8StNT"
            b"4PmbyBc+1w+1OSkFYIyS57C4DczWPtsRa43c5k3g20p27qshcb+ynI871wakVI4bo7Tx8a1RcNRYC1x6IozxXAccECIKcW9DEtrX"
            b"Uid0nMQIod/O9W/n+rdz/R2daxm3hBw9r8xnnWt50bmWF51redm5vqrHN5BH1bluQqrO9Tz+t1r//mrdSqGDQDZuQb0FpC3ngXAa"
            b"FwmkkbFeMeakstRA4BTJJO8s0myTGi75dwk2f5DFUsrTCx3nFjtWJCQIP4GwiV3Yy0d06Lmtf8BGnlsPnZsKomVjeJU6e4RMQLtt"
            b"bxznLyGhg8zas2I+nyJ3iaTCLYijP9tCabphwqSTpJMpUKfpNhtu8Er9uj5VOkVOwW/6c+c18cn072TUsY5pVzoce3IllUgQLhJu"
            b"5aXqyV4Yd92OC4vu6uGnADe+Qyv5bYvuzylyt/RlcayQPMbaIO+kNcw4psEbSyyzhHGtvYxfpKCMYN4JIhmTwbkg47fIqL9KBuim"
            b"A/UllOy5TNtdTYSbIkUPVIISrivVhjMjjr1RO2Nx3OshzHgkdWeCv+SGVMpCw82YUT1qPN1I9djsQVJxTlGrEn3sTvp7mQOLhzjH"
            b"DE6RZ1lB8Vb8lXRfy+UxRRWaH4wr19fW8iW9XznwfYGgx6jlcUx49coam3yNWmaTRzNNIPbi0a8i0HcrAsUqlelYbzmurXNIKyEJ"
            b"IBT/CXFrYxZDwiMd/zqZNpRKh7lEQSlvCFBJwi8Mox/IsQKxKOkBykbrW5ATdzENePfQo5sVyS6Hv1O127yVJ6LZkETwFgExdtZF"
            b"WRtfgCXSvdnAEkLuSXT7NbaeiqJggpFiZ3Ifa/v8+pvAC0xxuhVPsmotbilbiDKvyuDqyd4groHOuNdiY7UWWx3IJ1o8oxLbLwzj"
            b"LgzD0iC5IAqwD9yAIRwHig0yRnqMrImbMhsc5Q6kowKotQDEc8cxDU69xXmDi7L4Wdb7YPZ6TPlYF82jTtltkbJRRqyvqqf1Mt7r"
            b"SUHPdDBefX49F5/y8NmQxzLRx8rMhtg7A+wUC0PT7D7d/lfouKGsTZ97q/E74XmYKEakkExDofzEv8Gi/i1Rm9tHRSmZJ+d0wSXu"
            b"XeoHQUqk+v9JslynxAISlhVgWN5IiFet2yJP0T+ct16rXHqjlfpTUugXMD8+h26XL24rL/c0yxCnQTCjvGaWei4Fjsk1ZlDNgGrM"
            b"ubRMYw1OWawtYjYVlpgHnlBv4V9Fuz2ZPd2cek31fshqBbgPzbqvpHO4f/ImUFxeyR2TkWfKFQ/VKFJUGosduIyTqLyYiaVbVlh5"
            b"Y7/h1LMcuhr5HYDQ3fmvErE432/yqxJoOwoIE8ex/FQqGt+wXFmJf3ee9lSqklTSkq0L9RoAdyViWbANjabQAHVrHy1I09OH99oN"
            b"v8Ozr+o3GA6YGcm18x7HIpzgBGpzhmudONoyXpdTTMZynNu4gEiPdKw6EHJGe/ILgv4vgaB/Jri5RJZDGu3MOVD5MRa6nJZJylWn"
            b"UD/e0xJF0n9F5Zi1iypNDxB4+3W9C4OG5yzBcRHASxT0lc3ICgWNjwUBHyjoK74gP3WHalZgZVPyuzR8/9IQdwAOIwheSnAMOEeA"
            b"AJhwmsXvpeaMUKeDC9IokKApNhgpojy3ELT6AlxFjUT7VPzzGvysLoF09aVeA3sboF06rjoR79Qu4QbuufEMgRSwwQya85UVdjxi"
            b"BbquP62C/DqB6rzNHO/GBmnazQLYjVeJaGa2o3boDQBIOknevO0N52dY5KfKQ7jKcbhifWx56silpJnHkUoFmVQYtZUm0cvKeWs4"
            b"/40A42+EEW+JzDGpjY1py7OAdcxbTCpKSMxTkjJEpJPKiRgjtOA6MK2RVJynFjPTgdi/usN8r5N76w/3Sb83h8HpNq9mYaU4Rnvn"
            b"VjTNXtKmq6xfhnagWqM117cR4vlS5O2W8F09otuTv9uzRC4rhaO9oXKBsFiJFn1ZhxmeSb+zBlxGqjJxPqTDy27FFTPj9K079Swr"
            b"2fcjS9ZF6Ctj6N+O8vOOsgNtpBEglRegAjGCxExKtXOYm8AxVTimU+eEwVbF4jCGyFgdehQk59L8+kL/+kL/+kJ/yBc6dQweZegL"
            b"W2h8YQuNL22h8YUtdMeLG91F+Ui2G1SDfm2hP9vbyHGpAhMCY4mCE1xQb7i2HOLXnOhAPcXSx6dguFU+KKnBW8uM8TG9C/hBW/uX"
            b"CpwXxfLnb+1voZGvDDJbPl6MqbhoByf3RntAPK7uX9Gie9pbjqOqdCzrHkLLgqBFyFHxspJBVYe3gVwUktt7uLGnmbBQJGRFl5AV"
            b"XUJWDVLZ0yW2yVj7DF/Myb6LMvGntC6/UdFyT1/OeKy5c1TyhBHjQgShmEJJq0E7yQVDVICIgVzF4tMISmTMdaA1xZ7+x2TSd1CS"
            b"qkGoeC9bH4uJ400Kt5kZrYTRU5Di9cWxdfhpECd3izj5A+XeMdr13ukRvV/u4pB0C3L4h0TSP8uHba6Rjpf9yFeGE+PG+9yY/2qk"
            b"P9VI98gLYYhjPGnwMoMDRhiElw4xZZxnAbjXQkuEvdeaheDAYSwQYK+Q/Irktt5TDxvqwXRtmcfY1EPylvXasX3etbi6jPpq69sb"
            b"Si4am927Llif44dYIF6nEgWz7fGxN+6us6aTHZOY08tzej/zaQuE9egwHnY9fOs8rVeA+Zwnn6sArzBh9VnPwf7kWgTaDsCUqO2w"
            b"w9oCjc4Sm5S66vbHZfH6IgfLK2Wv0q4kVeuS1N7uE9HGa+dgMnP1nXCxqsiXIo+fsh2GOp+iPp+yOp82295FPr2YeosT8pJ+ufE3"
            b"m36/MNsUyy9tZnpMIeVdxFF8x2CoiVtgra3QFOLGN2Zdp4KgmAVrkOA8GbEjSUF6TiTDv07sF07sHyDg3pzEXBCEceF/tfjQLueJ"
            b"TBNTeBTr7ZlSeAO1LskJEx3aMusWajmM6plo2TEY86Gt3ca9sMcYp0eQxXOKR4Y8xlO0KskPVFlnt7yJAMcCtpIBrq3im8+cPmoJ"
            b"fXfuxNkzR4s5PAmWyfdKZVc2mFlS+VzsJsODEi+fSfFWhhac/Vqzf2r70vMQlDc8GIe1dyKmay91YNQbwbWXDIHCzDACwfJEkgkU"
            b"C2+cwFwrZz8EWoVrz/VjhzLf919v+l8Yrre5aGADwMxU/S1jzbR2D/Ipt6vpAfWfTzdOP3IAJcOueyvvrhId2d8bD/flyHBp9i1R"
            b"L3vVnzy/eYlMdr+o9z+m78uplfM98xrmeWwk6D42ygfMaR45LPcd6tr4YU8hfTXul8akyoeky5Okah08MV6bye7uE/760VkOn1yD"
            b"ChD148ze/3dwa7c50brN8LIsftmA+DLVRW+VSwK3hHtCmCAicOshcKOtp5Qmr3grDZMBwDPGRcDKWIadpxCEQj+qLF7VxJ/o4DbM"
            b"7Ds/tb5W7gvlXIgC7g3G0ITt24lZHSijV3IFK1u2idED2lsRfNEKfm70IDbLC7XtAxp6wqLffFNHQZUWTvxOIcauLNoAl5B3R03s"
            b"U0RfSPWMfIroyzpj/iViLu8LEnyk0hyqy0C4k1wLK3igQtvgtEA4cMCKcswBCDLJmBcJIxiyyHorZBDxn7EwVv9Vglyv59V3VbFu"
            b"7vFvSoA9ENm6qZY1MzvfmDly1rR40mP4aUpcK9P109xyarpe7gzeEbmkPmh6j0poeQ9OGBPNp8EztG0JK1zfN1W3uMDx8EdpeA1Q"
            b"OmixvKZE7eTXK9lxeSk73hBhq4dVq7d6OI+tTDKrJ7+aXN+tyRUokZomWWOMtIo1sPJcUWaJllJxiMVv/BslPDDt4uKhORMSY6E1"
            b"5Y4gKv7TQNd32g+3xcTnSNZPAteyDadVFapzNfRhLgiHIOxop3bRy0BTyP/UgXktiUjPHi+TjZxYubKK3NsIn+8HDAbH8wNWNsdf"
            b"AnN97oW51vfaGbC1Yk0FVa24CbwjLXSpv7ef6Firh+JN87Dmur7O57/Q1jd6wwEEthY0QiSgIAymzJMAxFmjiXHMOiMk4IBV0tFN"
            b"IgYYeW0s1lwC5V9TvX9E6QC/SL3wIPXerO27DfNADcAX1AAiT8mCAr7arvAIpz16v0SekgINEn8qKZB3B4CpqudeUCsRdJ4RG1sB"
            b"jTI0beib7j0LIZ74fgnWiXZ/T9E6d7a8AUEO9VzK24b4+JnSJbLBrGPaiM6mFvmUmMX/Np3larY4ZT6k+HKb07YXmmOnjkUlDFeS"
            b"ug91z59RztbCNRsCjtcQON6tAc9UEOYUszP3n5i3Hv322VS079BGf0POYF3Ff6We+v0qnnuMQ5Be09TRAcOJps4Q5Q3yQjLOJNfI"
            b"m1jcc01jUkzjGw/SksSNUN9kE/FsSbheDz4skz6tw1fmB90+v/e3wRU2TLxnE3SDozE3McJDUiQboEG2SXz6UVJUSuOVijsbxdS6"
            b"/k5W9T2VzHCnLNbl541MUTLsdtG0VTDv7X1ScBZA37rYdMvJYtQyShHb0rIX4A+z8nNPn6IUfLLJ2mdkI64cf+bVsxxbMIt8r8Tr"
            b"p5sg5UmvaJ6tZHnnaLtzlFhIZ/9aXv5D2XfPuLGsQ8xLSx3C4IxiQjNhDGgvTKwTKLNcBmYZd5IEgxENhPLgOVGgPPkC3QU6B3R8"
            b"Two+isacbWBQbUxdSVKARr1aY1O0b+iGko7kZSaaw3knNgqZ0EFlKfDkjuTd6RLNRDGFpb/GuME6wnGF/e092G4huuegkdUZ0RYY"
            b"MxuqqCV4AGqk19khMNEH5FfYuw49D3UZr4TK54QMuTUnatMyVik2sso/njWajJNW8gOkxDcnupLQXmghPLLn+Uw6hkfYSxe8JNYx"
            b"hIxNKuWcYqdTKgMlFZZWM2KZ4E45YLG6pFjgWEDGuuGrvIB/aD/hk30h722zJxqJeK2ROFM9xF1jtBcMvNUTeCh2S6ppJCo3oGKc"
            b"zZByJRIDQ3JsDoznV6U1K9PckDI275MMv5gtOP1snNKuDsqjw3xZmGMhThueU4VsfmAJzxA2LMX5djuQcXlUDC5X+G6X4SnsjVU9"
            b"ATakblal7tq3klUA49uwt7nU19LF53SL78V4fzsM39xh8EgoiyVYzJPWukOSKmek0ImTTHhcQDwHzbH1DAjmjoA0ThNrCQjpdfgr"
            b"wHLo03zkyUbMqHQQeuZBfpWAqnXE3uJKLDFkQ29jz+FoMjrrgqFQ/OL3FVrdX5jitePHyIFUAhzt4LnFei/qMDG6z2dJ790rMPBL"
            b"MfoS0/0m8jlSxJsaiqlT9oCvsdL8KtyLvsYlG7linjRhiS+uWRotc4MdHA324wUUW0wcnOOWOxKKn4eJ8yhu0AMwguLPQDzRJu7a"
            b"UVCxCjYh0MA0SAVG+bhvZwEjTbT2lnJjCI5b+/8YDnj8o2NDDCkuDZzsYOB52F3063Xm4j1ereQtJBuwL16E57hsJDYmuhHimz5S"
            b"SXQC5FkMlr+CMgVs+WICimg5Q5hViIwzvtvT47Rlx3Bmx0nYLIN2MfkcEr0LAwbCUv3+oAGwRBfIC3TBZsBQNT/vmjOsvMpaVFlh"
            b"qv1Cg2+mQUOUpE5iE2TMerGAU9wkoQaJqWWOGMks85SB8QKExgkSxhB4kBrU2qv3v+ZVdltJ9ZZ67W3a8BcYlv0MqdcSUgQZNvhW"
            b"RU7u2qdz6Jb8X92toJforccuuU+qw5XIAiz1Za/YvOtac99WD49OCu/PZ/P+CzqyHjnQDoSBWFYiprQIUgshNfOCa+q8Aq6MY4xa"
            b"jwFCMB47hJHBxkFg6issaJpacY2WvTaneQWVVa88w+iNanBqXnMXKduVSYfdVpe7boB4F+qBZP8c/RnvWZnls6V7QikmPTOXT8ra"
            b"Z1KznRJja8IwsH7LWdNPOmH/LsVsabn4ZzRgId/FygLlUiD1qL+58Ro4q1kOnY1jXXNyNlagvClBu1d7S96dW0HJNReiKWGbJ3+L"
            b"LOxshvVh5OzXcH1jcpXxyxcw40TE/09FLBJ0MgRTwnGLY6FLQSvnEEHUOoRiwpbKmkBVkESZbxrX014l9mxM4lc2L+yqPbnm8tLD"
            b"EVteMXovSqiK10sz+FBVp5vE3yq1Ya9nR7PcruSbl9G9BOvKS3dUa22rXtre23mJf49xfLwbH1fIdFUp9t0m5bMidE7bvXplRc69"
            b"emWf41w9+jEtyqbYZFsf50MuL586qMdgqSfBaCS0FVL4+KWIWUoKsIKLYAJwqYPzsZCUTAgvrBISI6sQSsZW3zqo/2B9ia/qS7hR"
            b"X8Kn15dkQXTt1Klv15fddH2tTn1ZX14jB9blJZ53Tcfy8orluxT6hgfV5dUbXFWXqO0BLavLh4NzSgHiH9CjZHpVXsJleQmX5SVc"
            b"lJdQl5dwlpdwlpdQV5Tdk5+Gkbqhxt1Ul+td/lcOyq+rS840ciwWihaLVEAy5jgjQRMJILBj2BqHGMeU6kSmR8mOFiNJhJKS6R9F"
            b"or3K2Z/UEHgtW9jPvqt5z6vZ97QZME6QzkHaEDyffZ+LwCAJ1vYZJovAtAPS3RRZrM0J62xeh4FVuqgUFZMcX4IHSHk5/qi8vBMO"
            b"dxCHTBG1cvd/fDT+U3ipZQ89Gwp9Jft0GAoltyvNbDJHUQ5LaohEIe6cCfWIx021N0Zol1imMn4vvWFWeerjXxAOMmbAr1AjfGY0"
            b"IF55DVxrRE/agTfFCW9L6c1IPVMu/iTZTUWlFgjUXvRwLkOzkj08BjxlR0uO4nlEdIqi3Ve5u5x1G6lEWbsbNrCqSGU4MBnZEXko"
            b"05BNC/EQQoTZEbSIxqRgIAzzXj5x4YiQP3LO4HRPuJcOCjmsvAFnAshr1cV0p0poY5P9VO+QYcXStT41zOa1XgCv3Qs5q/I1Z41p"
            b"dl3X1s/wQWjie95unt5wjJmZZN369x+vkNhAPo/i9Sh3aa2dSOuFgNXaiexbtBNvQ0OxZPH/ggXMqJWaMnCcxvqX2yAw4V4GKqwI"
            b"jFDwTikj46IiGDDpnOL0VW91aunwhqHDlbDtnsfW68ynLjKVpw3M1oTTK2azXumXllXVKwZ16y2lEnQnQWaeVnKXJXg3oy3HyFVG"
            b"3XCuZZEpGTQXPb1GQYo4Vo4d6NAE8a3kF7zTDB5MDtN5UpiUdaW8usD4CWQBQu0V+pH0G7zrwad638Im/VY+x8SGFWzAkedIw58a"
            b"wfp4g+SzCpLfZNvOW7ai+/84bdmliU07ViqZ8ImNDepq3b3iva6W6w7AWRVribwJ4A0SDAVGFVUIC20cBxZ3+1oh6aQPscRyzsb/"
            b"xFSHUhoEqnlw/4TF4DyjfcxiELZy+jgR7wbXT/0It+r64z6D6RQxQnJa8ZlgFjl2o8lgnXDTk5zvjWFCyfC23SnTleW4BvK5Lwm3"
            b"cxh5hPos057KLYtU3CFSTZJII+ZKOr+uV2LaFWFo/nDLZ38curRv3qvqr7Q01xqBR8Untl/pZKL02R6DHhsQYHAIhDiHQ6zVsI1v"
            b"qJQOLBZqWkvuMGPCkUCps5whSJx27FUM1uTvlgu5p/FxU8v1+4mjtVHWLcWOQU6btrfsC5mgRXi16D6htt0ra8XWVnYVNg1VzDGD"
            b"E2Rf8y1xbQzW8idZkQ6JSy+G8zOL6WfOMfwDEyPyKXLZV0LaH5HLfvboVx3k09RBYoIVTFLiiUBWMRkUUB83v1hrbFWa0LvEn/QS"
            b"eKwLDYDE1hDCDHLIxu3xDyoQ0Svc0Rp01DGCikwd4acMUJ5atBJ1OSD+EOfoaxI3uqUeLnvDznNmldqHpffLMTvcaCWvP8ml5z2o"
            b"oEYJaYQV3r/Oas+79BpIlLIb7ZuWyZcrjXLoflP4ngj5YxGDfJLc8H0PaFR6xk82uautLNvMqPbOI+mtqFntRN1Ki+DKs3Di1fpH"
            b"bag/Tot8G3H0ySWi9UYpsEBDTF4WsPBEExe4dqCAo5jcjASnDEhNhU9W1BQSY5LjQPy3GgV8ykb3kjGk3tm/3YYSLbaMLybNxz71"
            b"MxFHaPdEFe+Dg1Yb/Q6vs4QQjdiegupBu4oTLqSlvZ+5uFub8BwGehIvLz5kDtyQSsAYzN7oUPhruqY5Ol1aSyJ6uvfe4UcP0usV"
            b"+ghfoo/wJfoIX6CPcI0+wif6qEm2FeCoe/IJ6KM/tVW/Ief/LZv42QjGOGA+MEwAOJdcIR3Tt1UCiI6ZOSQzlxjk46Y+1qVMMS7j"
            b"PwJPFE4r5VfQjG7t3t+V3r+aXd+bho8CnA0mic5QSbd38jfRS5CZ0kBYDTLaRzS9H+ox624G1ytp/fRB0omFxDWwQC0n0OlP8WJi"
            b"36NLZ4AmsprUr9ADc9Pw/AnzDYZ62aStBffsbdB2RPzgbJ9I1fP9ueJevqhyyMfm9Y9F/5+hp1ZOLpCrgpMd3z6DSycXOJ6xhlnP"
            b"BizVNmPvEaczZOkIPz35pn8Lt+kNAafG5LBZGb6S9XR/Sh+s1eCVDthKqqRgLIBnhhnjQTusidQhBOo0psglpj+J1b+KLwLSzH2r"
            b"ysnX+Ly8LQ44moCT7+wCrwj/r21ehlY3X7u3zCxhpoi1C2/uT7F56Rvc53XjReYvrH+R5sGgqstnldYeTGgjoHY/GeAYD5+bDfTV"
            b"+EJ2k3mzXbwBtJ75f6/MXmBp9gIXcixwIccydwE/HF6ah4fZC3yOedefbSB/iFPwNV4vnmCqkaVISeBCe+xckNzGstc7hX3M1Eq5"
            b"kNzBg5TOWi8IQ1RRgyUTGv0bygH3KFs39QU+WzkgAfb3PMjlGUwrLOkofMLLhcaNGpE10L8GodK1rgs+lV0WgrIvZAfY9tbxJyUn"
            b"faBi7I9nfo2JLrIy+ZTf6wn+wFkxlur8kcj/nJSwK1c3mtQv5U8LVksePe398ZVpIps43baRvxIBn9hDITF3eq8Cjt/1WB8zp5AA"
            b"ZKULVFudVAIEptRorylBKTNREAyIo84Zu5ZquZIIOH5XCyjrIjfzDzRarrssc5Aq3WCqLW+hJNptmLfVS7JdG3o465GX+NQd4Hz5"
            b"JtpiLvlPZA7YHbr4HscGAEBK8zFq18omezAaKvx4eVl0ejMGkJWB+LSrQHCJjd8Tut+jcpB6y0IAZcSdIO9TtR5N7eZy/ngp579H"
            b"sWoy1ypHj3L+q5nd+K8/J8091/N/lc4+VSaAMEu55lI5oS0lNGn5EeqpDlYljT7MNVAhpYnfIoOQM/GJkYQDl7GC5H8dW+sm0+km"
            b"Y+uWn+ueq/hwpuFDvkWzas38aqwTJVe2hR1gnvQE/DK+2zfgexHZ36nbvrD5fdMvh2yE1a0BgLed9gHO3uNTWPn4p7VhVfZujADS"
            b"WwyWYCBsL1abg1BLzzikA+LHzPt0ijjeeRb8bOeeZrFthi9vE3+KrUvNmx740WLpbMLje5RjPsTeegq9nbMHrmC0m4dg7SfYANBm"
            b"ZztBYqdU/ynL38r3T+vV6ohfdtafYGcRlmhWJslYe088UyEuDUKhmPNTxUsJj39hznqsjFaWGhe4iStF4ARrT3+WcL9Y4zvwK3zH"
            b"c1HY97q6H5NxnfSa6St7bzEId93u5U4AaEu92pnroah7IzO37rLusDLirIS7d12bqVV3WgIp3sr3HYmHF8eUyHxRcamlpDloqhiW"
            b"/jVH9u7ek+HgR/VhnzPCRhjw1SsrGPDVKzMz19UjVj368ULaRcTrWX/3KyW2r/u7IDhJmdnE7y/zBhBo6wEFE5QIIHUs04XDwqaB"
            b"XTA8NSGslNpYyRUl/xjATnxC6v0QBm9h0jeqYl+MvzJfbpS63trThEnUO1WN3d5bLIwUEG+OqHpLeyHVmaBvb5t+stHzanj7HCY7"
            b"We4pTyO9eQnLJUSvi9BSOlLARumgRSmyQ5Yf2RZwwZZ9R7YdcmLz+FNfeePRT8+2X6DC/XWtXa0IJ8ZapbwSwBDSWmqNfGAhWMO9"
            b"kjgkxQ6nlQIjQ9yuSyBWcyQMga+YpQ0zsWXng007HxcQM/xe92EAC8AOFpgoGGQllV18RXaaWnOAQbrYEs2URDXmrYpvr74EZvCb"
            b"3FkXakfX8lk4l2VNAKwwbw+guO0vlJC8jABR7HQWOBvA3ek/hiSsT1XeECedGAWHipeqBXWGAyD9VaUHUtayX5UKTXsIqVDbmwvj"
            b"Lho3SfwpppybAxHyvPa8+kwEG9IHLKFt2yOHPhjLCRwXBv4kc2dTzNNTu3uKiwH3YSTTPsN77I5x655uQOd9PNc+OYy0d1IzuyFL"
            b"w1kFgp492VAVX9zkhro7gvruCLvTHalx0KvuyNZCudEdyROSfF1jc/wbuyNGxvqaSB8kBWkpM0CDgCTuYJzlnHJPrAUGXsSnwUgT"
            b"KJi4LHClsQ8/cGW41cVeYNHehE4/6HAv6ttBYm3rgvC9qp331k9UNAY+MUX4yAp3TXTuGTzVpoCQPdkeqZaMV56vOK9shKDz0tne"
            b"1+aDyWKJLCvvYbOjWoff7rLSjS7BgDCBFgk+P6QE5k8NlGB6sjXVQu6thJUP/qGlAPMf2ixhk0evmiXPGii/Kf4LU7yz0nPqWEhS"
            b"FzFpI3AkGMaIEB6x4FmizsR9gVPccc2FZTS52IKWTir9b2unv4WwgwcIu49op08RdnWz5T7C7rLnchNht4LOXTRU7mm13yFuXwHs"
            b"rvTQpwC7OSFzANg9hTcjxZ9hm1eyvyuE3ZUg8Aph19BVOq30WhK4I6q8RNj9qqQ/7cJQxC0L2ov4R50gyhKI5yEmW2+EVolxgrhX"
            b"xlHFwMX/xJzMDEM0SJCB3ay1Lwrt70DNTUrMK3Ghs769rS50t5b/uLoQr3HEN+WF8K7Qc1NeaJRi+zR5oVzYPJIXOirpR/JCcOrT"
            b"3dIYeobck3+rtc8zu5+/kvb3h8h9ezqFECQBSbSVVCsEhhJpmZPcKO0Y4cpqHyDhlkWglhOFSYgXiR2TyvL302kpYD4C87vG+LX5"
            b"IOcsSZBspXo3EM4VC+S4mJdmv3Md9UF/UewlHtl3zk3qT9eYQzFmlYsDPz51e8ElKnUOMZzSuIfAQFN35oj0U04oG/QhWQOXxSD+"
            b"EG3BS+kN0GS3qIjtspa0j2ecj5eEj6ewumJl/iiJ8kPBfGsBk8a2p/Dpdn51+wyWEr1XRe1Gls574v7ZWaaej2ou9ick1p8Hrftz"
            b"ALo9twqkvZJCMC0DmFirakMUU9Yp663QhMdLcZYy5y21WMccTMFoS4yXUqP/MGju0xFxr82C0mtpzVdrNxzIOzIitpJNnNiw3hJH"
            b"FHctJDrXSXTcrfaAFFkWif9n77oSW9d17YTOBwvYBvM+WOc/hMeiQlKkLDm2d5KbW7KtGJJlxYZAYBUqVO31iKrvjzwf3u5Tgkf6"
            b"IFUbo9cHOQOlHMnP6aAbNkXszWRWxx/U5AVfljfo4Gt5fJf5gpTQZ/EcdzDRxZUCVwkWV5JysvJQl80zY3d1/J8coKp3XSU80lbn"
            b"/cPNXR1/O3f1X4Gho4L5AMJQq7VlQRvMLXOBQcDgY+EbXOr2AiIUSDABOc1RSHp2MZ9bhH6c7NELBnL8Yboek+rmHJIn6MKP5nYP"
            b"sXlnXYFb2kT5kAlBsCnh4XqNcYQbD3w1RZVp6VQN7wEv+5Zhe7lu+QxiIufyGo87HbuEMxr/2f+KcgIJiX/pEvm0zBG550Y0Nohb"
            b"fYpwrX3UUBA3qAfUSI9V064RuKukSGHw6MhBnD/84GjvT9Bo0E42wRgklGFGcgmBMmwoc84wTAMJCKzDHjBnacCHmPTx06g9J/F3"
            b"zIF4g6X7pEB/+jbxvGDROnVa/SIOyealVsjD8eBUDa7jldfXbVdtTiW/IMdk31e9tWcQrhEzXSW9zfjEygeU1f39wLauoHOyfOBX"
            b"NTg2ukHl45ZgzKhg9diRLvtM7g6i3E02fqNYa3BUX79u/UDLm8YINRNONmtk5XPK0R9Sup/7guBK6Rk3viDPqOOTIRHxxyjd36KT"
            b"f3Gs9xqieT65JRuvediiEERcGyIvPAeahnYaY69AiVg8aAoYY8OFAC4Q8JiBeQjeUK6ExvaX4ysu59BTqTZ1XX7tZUSaA5Jh4IfS"
            b"UxOHHJoBrY5CiYkFLBmA0jrMRelSbJC3Ro754CydD9nL+PdHzc+nS0JlEenfeS6H0BRRXh1zAqKBuo9Fq3LYwkd8KtliitNlvKNV"
            b"NNN0W6WCKvGidmNSIS/PNTxweKhYxKHSEF02tqpa/jnPvx5TAVgD4pZ4zOJHxnCCPYNY4mrCEA069aspaGeoYyIw4kRcBALz3BCt"
            b"LJGvxVS8RB/upB/yEEFxHz4hl1U3r6w6joMryBFi4e+t5GLSidTLPM2TAE0UPzjEpYj0vpQgp0kKckQs2zYidC7rhzTl1Q+FQyMc"
            b"cjA6QbwExTyNugZA58SSI+IP3MgXdU6cApfbx2dUiaCqKtvHUHmJQOclIisrEVm1FHDVUsDXVImq9u+PlF77oMDamp8I0cFLarhS"
            b"DCNKacJ1Eaqs9ZZrzCxXgOIvnBfS6ljYgKYYKNXeY8E+Jar2MTzDuahar5TGe9/fVimNz1WHYFmSp295CZZzQESfJ8eLSFE6lIsH"
            b"x3rQKRwixaVatVdMm7nGF7m0VJ7ipXisdjmo7gxr1mMUlptqGlov+3rQ17r/5m77PazWWGSNTUXW2Fq87Z4dvFbvJSt0oOK98R5k"
            b"8H559s+4AVddywU2cMcL+IWaa0CkCyIpR6WKS2AwSohYjoVYjDGpGNVOsqAQsRocUUo4FFdvngTtMA/4bwZ1GQ17xaIjPbsujxmw"
            b"DhKqjsbFB8m1ptuIO2hsioofqUoebON4rfP+tozKJ5H3iAtCrFbEqFoLuczKofgSSW2k2X7NaGSkFle3SvGBpl2mc3zD9ZLqVPix"
            b"IZkjCzXtSO0edmRLYD6hEVe7L0fzxSuBzw6jntBXGzUdZ4iBpQm5Zd/2mbnyGj5RVdt/d66v9ivRtT9hBgUUIcmRBaccxLwvVUzs"
            b"1EiJLaEx3VOHkaVKx+LWsHiWHmnKpOeOO6Exfkd5O79VfNii6V5b8TTzX5ZuP/F27jPn0dJuBj+4pjrcsshm4IjC4+KbF8YSNXEm"
            b"ymZ0WUcNiBxn1Va9h5Sau5bvWWi+B5HkFJKhuJTwkfbFwV8jx1WSaXNT7KNq2ntT9bgvsE6bSDVtIg+T+INpE+8fjhJzP436ofrF"
            b"XxRqf9vAKfUvqdWCEEyoDMHq4KgWIY2WAnLCKGOFlDYGGZp+MoIMijV5KsUN/60V950ibybfO1PWuVPNP1L6RYcq+ZV0uHv1/Ilm"
            b"w6BDkePKGoBSvk2ueFPO464+z/rF8SdvRB5mlwIvh04/K9/S+UucNLBbzEc5hxTICB50nvuCO71+iXyy4E6f2DsNkUXDZ8Ny1VsL"
            b"K2OX6Wm2Fgc8qA3wrlCGtzTeNUR2G6QRlW23RqpMkB7Ce/9K8HeV4NpRG4T21lqiHPIEguTGAeLUYySF0Yo6q2OFLh2SJt44hOQJ"
            b"HqySq/WP07y/kAQv6t1vzWjcYbF6p4+SYWg9qR+rwl9l1l3U7b8pnT/0GJ1lcgQFtkYp77Ur8UFS/iB9TKrAwdKHyEodD61nXXI+"
            b"G8KQS+8kBbOkQ9jfNMenVc6+7HDVulRU1qWEPbZdKmHxM/Qlcfv7LDyovaybDZardw4VUpi30hEcKt+l6rklkMN+h2mOCgN3u8rT"
            b"tNJ5a0Iqnbdx/J/w/b8QvgdrgqSOCmUcRkFp5RV3ySHYSS+UtAG8oNprAdp4Gm8bwqnU0UmEViy/FYdvRuB7neLyUVa+05Tvlen7"
            b"NX/6NPzHcIN92NviTVGck1OFusJVf7lhAJPS375AgGP5u3DEaOzxPYQu5XqyfxMmQ8/jDW1MUlRkpYrvf4GZ4FA6xxwHqMOKdIEM"
            b"l5B1LnmTPUdeIx3xtEDEQA65rqPXLkkvwPMDSHIFCNZNJd9OhTvQ38Ah4Qz44JhOXWerkhOdFFolUoSymFtuDaD4W5twGAyU4AGU"
            b"d6Aola9VgFhz1XaTGlfE5+XwSSF8U5rgmpnTjHLVAy96R9ClSSof6U9Arx+Dlh7Nomch1q4LySsx0oqf5ai1QK7EIVpjpVT0onxn"
            b"xocCkHf0rxyzgiHwBofAB0QE3hARuFO5IQcJiIyT+iR0AipR3vpxqRplRRSTW95J/QBK1n5Cu8VOgLBNRTd+SEk51vev874Cmngt"
            b"LCxQxhXQmIesQkTLuHBXFkAjgWKeMhQFpDVOTzqOmJTBYa60Rpop69lHOQPvpWo9xucPkfejoctBKkdWMFDyDOl3uP6eqDTMNGXI"
            b"kx7zB6+hbTQ1IuXSokxQl1+sUXI/ctb4ykLeV9x10TachMklmFWWRhVxebxTer9lB3ZwNaLz3fK1zH/N1K5ge2U+nxemsNzWplQ1"
            b"LzN8gRJWzmlb3d/mhaW/6PVh3cyzvozk6jU9XGIpnDWNG05CTU/YfZSrJX3juZy7wQf+wtYk/rlO99Mu7yo7dezyvpPbcHkxzzDC"
            b"wsZ7BmKESiOpj4t6BYwbpigx2FmwUhBpuYi3EAgYoSC9x9RxKbn6m/ldaxx/s7HgNeWEGyM/Uo38Fk8MMe8S50lfjkzjMFQX19PR"
            b"XYnFe7aeRD8xGszHLfGphStGamtd8zaHpeYtCL7F87UOalvcKSL9IZ8c9929E5y5dCwADV6LQeystayKBLX+Wu1xOh72NZiNKqWv"
            b"o74xXXjH3x3vJH+jvo+P+lhM+yR+/oOhgVlwlGBAkluGgwCOrQkkGKRFXC4IDdSk5B/XGEEEEBp/1TJvvmT4ynrhfLHwZTfSr2gB"
            b"T8TLxXXlh05TISVHgdaFzAK2m+sKD9cKdxYK/Snko/WDzOMx8WInWmkC4eoWcfAspbsIcWc3jdfOfg+OT9S6dHRKVwL0vgeMtHvK"
            b"sXP4s6ziJ5SEj47QZ+yUM6DF2JN6rhV8osy+PPxdNfg/qrS3xAoWGaYNlcx5ErjCMlbQGAz3NjDNKZWCBcEcM5b4oCiiGglJXMBA"
            b"1Ue9SL+oTznLffLg8tkW1vtE7VJh3fdwDoV1P3S6UliPHaNHhXU/feqPfqiY9y/CxYpZjgjXo6K2d8foGcnzGnvKqjncmMZ+Fl17"
            b"/8wsI0uj5ZNOkDdoq/fRaS9x6Sc7eJ/21n0ppry7rba+a2yaPrXXqdLnDnjz2hqf1NZ4mt8bycuKMr3rZI4Mk3btzK22/gUqmXVu"
            b"r82UTvor73RFvV5aG6AuUEUFUSwop2LKR8pxob22yjtOrWBeOQICjAgZN4Ep9gIbGbB/BiyxmlGeqandasg0E8hJSQ7nVntnHRo1"
            b"xzBsZzvoTpA111I+RDHgu0PLeJzM21iIdo0cxSHPJ774Nj7s8XUdDqLv9XT4hmVweWwMtX2NJWYwtmxOLbP/shksWl4+f0p2/Mya"
            b"6vPz8QfsQRj1k4dnFSh+qmH0928+lFbCCCLxuJXwQogECzomMc6UMzxobp3nMV8xDMxpKTQ45WNRC8LE6teqYIg1hFgUkOeWUvFj"
            b"9XFm2F640l89Qz6stmuPBHdpb4MzQj5sHm4tpWGOfMBPIR/wota7r7bpUqOTA4SZlaj447xfeiFV52PkCv6p3JRqRX5LPneMisBT"
            b"VAQ+RUU8FtA9lnzVw9egIv53lHI4ZjpWYAFZRuNqPP5PGU21Bi+cp0ITLz1XCgtqGULgiJPScBV8AG29/Jtwfbvx1UvZalc5cn+s"
            b"tvey2spn5s5SfA5QwKesNnzCaru4GK9Ybe9Zjv+Nut60HueEWZamWN4waiwmzisGwCXlgWvvkpguGA6IxKTnvNaECsGYRt7oYPC3"
            b"Wo+vaKX5veXNwhPjKVj6QhXAHOOd2+dOc9+83VgBmMn07e/vLOwI3s03lvTyOC41+L5HDUlrMXl8CY0/RWVFwSu7O9K6x4lSYQ3c"
            b"48g9/6TKX46vykZqrY97vDETSyDGoiTzTdBoCUc96O+aSJFcIIQYQSOyyw7nkCWKUhhwgipZjMGLl5PMcc+KUdxrvI4lgM7z9dx7"
            b"fu5kT2rjoS6X77/bs/qPlwj6ghLF2/yHOCXMaAgUYSGlpkyrRDOzknsutUcSeekACeO0cWAtY5AIaQRRaSmCT6lbwofMjysOWRFn"
            b"qz8pd5WBsNjExB8q5fS6mXBc/9e6mdC6Ah2kbiDjCRriA8wxwdfkgUgpaeWC/92kO+HgGJzuSzFKcTS8bTSZNQUpvv75T04yvx/F"
            b"n+2Tpvd3R5FnLG0pp+AB+Z+sYAHt4wfiaQdI1/G3P1R/54MqO2s+Ew5bhJTW1lNvWaw3LRBq479KE4adcSZoZbE1zvrABRMk6e9I"
            b"7rBH7n+56qzW5/Xq/HNu7hN/twOtVx7y8T3FySMqaygFMbCpIOs5opEjZZ/vy+nOXd4ao/q9eS1qmFb7CumsFzcJikk1PNt3nZwZ"
            b"WvbAlKhtv12KfXB2i2KaeloDjQuMpORwywUi92crmCyu2BUz3kUxMN47ELJzejj1lehQsum/C0q2ejhyhvgrRF9biBoZlA5YeWlj"
            b"Naq5N95gS40VSiNOBOM0eAwySCoCYG2MR1QragxDkn8UwFUl6KmuAXmptMG43uwQX0ea3TUBsg7p1SbTzSaGDiAE63IXtTDj4wp+"
            b"aKuzfPJJp0GZXHWaDDqMzCHFAIeJtc7le/U82CMFJt+eg2POWOXh0F0pf7djayVfgKW/cvT4WfdpG9YpuMhiKnI8GXq070lxz5oQ"
            b"36t9H1SrQ1c0qLZg4JcGVT8ADt5nLXFhpEUJFb3tu0Or6kK4TsFfsh9+k/kO91oLC4ZL4TQODIxkGmvOKXFIEcKDRSYu9J3z4LCw"
            b"RmFOvKMAFJym7xjZHRBNU00FGImLzUvBgQvhSV8PP6niRXklnCW7NuC4Ii165RlJNNAe3/wT8UF7PPN/JZedUC4fhXNZbiAjqXLc"
            b"NyfVOtZjsf4bSet2h78oqD9Rv2+hpukFMz9ZMb42fFU9+TvswDJILD6QlY2wqhXH+kHkrpyM1vmi2CeMvYIxWo7NGRFyP3e+LpL6"
            b"w8c3WEJbwbEceiNp0/ilROpW6uZZEo5XA7d68xFjbYldp3fd5uLVsWEy6o2Vhtwn7AM5uSEtj7zWmo2lAn9zpc1qUQrUi1LAFfGx"
            b"Gic7Ex9bmiIXxMcIzh8XNqrDPyc+JuL3GQeFjDbEWcmdEgZrneBoTEvMpbYgmJckSVUaz9OgT2MIkmCujftW4mNiXqfjR3X6fYJG"
            b"rzOGn6nCj20PcRA0u16AlxI1/SM72bJytyOP6142KX1LVxik4L1sGXuo2SGW/Nl0Ho6Typppcm1SiSvL4qN6SD7ZZVbJKW0wF9VO"
            b"bWunRFbq8StC4+Bk10vH36/W78pM4goaV4ubzZzl2X+yKeyh2guq/gk0rk67bPAufFaJCG1OmpsWcbXXz+VDfKlsf1OzRBAUgkmk"
            b"YMoEM1oZxQNX3NNYmjtPAAUA5dJvsVCxiojfKYFiCjfGK2bfYBw/cY3/CK/44tCuSssnGkSXJnt04+wWuAEt+IExOk2UsNXR98Sk"
            b"IkWk8paIXZiHvnJmxy7P7Hr1oWaFVWZ2cj9JMX9HIkeuM7ubhF9+j90wntnh6cwOT2d2eNoFOXXXbH77Mz3aP+jEviYz0EyZBCrQ"
            b"iMXykQZHEcUgsTYcTAAfsCYkFpUcUemwVgFwoJ4I64TF/1NKahdX2rC0LNJPOSgN6bMI3qGcAh8oKrxNomZKiR0Q127J2ZSodGAM"
            b"mKr9qrETVHCOLFeZIrYQLvBKpBNVg4WeU7OniPCiCZcb3WRP440pHe0B5CWSMlSq/ZLMx6dSzrqEPqugRjjBALd4HtUcrtk4p/Yu"
            b"gVXVyqt8nfsctTL6tjVCBHfpu+pUNCFVP2Ic/yek9mkhNcGQ5zhWtybeMBKsw6KAqSPOxQ+wACm15vF+AoxgiinThlBNtWXxzsKk"
            b"D//Yse6L7e0nPenGDstTCIZ40FOfNdTHRekB0PFyn7kaSrFNALfvERv3/k/s8/qmfo7JeJB9vgdrfucHZ9DhbFJWWIuxN/UyFiSM"
            b"LK8hV4TG9j56CVCZzWLTDg1I43av+SY4bm5XB5VdHVyyq3vgRnrdrq7Bzn2fnnGTfyf+zyfEjtIbppd7w2/zrBPJ5p7xAE4ZxqxR"
            b"XDFncEyzjDFNpBWBAsT8a52yXkspWMy63ikpLGL+lwE0+mwySKVDrZuj3BhZ7uN9xs1WlTJF1SiMYfRTwmojBEmXmncN5D4wR+RA"
            b"1ikfjELZkgyBNvO5RWa9O4WWrr2JofG2XTFK39UK55gogebLCQjj9SR2iIYcXdUcujiY8rgibc4o19+df1OKkajr+643k3dhNUgl"
            b"S0YuYY7nApOwJvAFxQFb+h6n4kpuuLV0biL/eet3rYSr6V2pkL8keCNKnh+1f98K2RCCgERB+1j1Sq6xFMmvWUP8NeMOuWCxxJ4j"
            b"gq3FzmHgRnvDU3UslEWv1YPoq9nzUvYcpjGI+greuDdsy0s2WZN9m3O+iWTITQiuWiTDQZFWrm3iFUSxTNzk7pjRmahlDjPPK/5a"
            b"a2fWix1wqoel/tRcblhOpzeWzwAQsFpqUk2F3mNcOXQzILtrsHy3br3tu7HkrEpcHaphGVT5EqqB2KwjsXq84bqRsQ3YtmlYZdJZ"
            b"PawC1jzKNtn2XwZ5+HfAhjVp6ri2YpYKLwyTyKEAnnghA7egg7TIUR8MV0bjwL0yRFNNTAjEeoat+WDP4ONunJe7AD1FbWDHOSwy"
            b"D56Yq7DPIutz9M7o0/7Q2OyMnoFHq/JM7033HLm/hzK4Gl6XFJeq9b2nINZbC520P9hWPotVElfuqMKjN2i5i0hobadGbzYFbcSP"
            b"1e+zoaOU3QZvo1A/UneE0pUcLVeu8zZdbBmE8fxLdJudb7oixZdNl+NeY4FUmj9kg5fhBZawqwHhh7SQMzuOlfixY5DH4uo4N5Ff"
            b"Jpr+Uoek2uR+1ljYPJIeNhYeuie9r7GgqbM01rA8/nAgiQ3Wg1UY45iaSUrWOBCDLBYBFEHCciGE1J4rBtj+UfZOKXv/I1oSDyUi"
            b"UqmZj/WnEfFlafWZRsQjAfWZ0+bct5MMnTaHILTq2T9q3mvRZsYFEWjMxlyJoDEzjGjklWPOp6EaMhg7pIWPJXaSfAtMcgTBo1hd"
            b"a+te23F4l/DDde+5oVjkxDHiqksdWr13CxhVrMlx5S80vc98jjk0ZrQVtLH5cOaM3QvtAMr18gOhHahkdCoqxaZb1qXIx72ZS5Z6"
            b"V0Q88yHKaWHGVyu51jiofuEck8EbHLNTFcwSkeWIn8uc943hs0ga3zEP7SZb+My8ljCvzDzJ2qbdGru1cTxUhXHlUVHwuLmQ67f2"
            b"5sP+aEE3/I6UOjWr+LxM2ppQPUmANsUtM9h75pRFzBgcnEaCYRWosC51LIDwWIlQTp2g2HKmAonV78+QKj/pU6CHsId2CdtTIabI"
            b"3rUM7rqfdFEJ52QL2doh3QwnJ6OBIsMx9kr+EwWtpugSsnMuhocc3C+GnRjY+gy8vYeOgtOL50hAuFZ+WL5svWdRCsoiOyvCbpOJ"
            b"aN9ZiWhMk282bwErSB3jG/2BTPSrlB6qrdysqZBk7VZp5ewEiHZrM+iEWie4VqHga0Ng0wQ+kNU+ozT5gs5tU2GKfS1xsUP7Qmlz"
            b"4ZPBmUDUEB5TIufJADN+VrUDHMtNA94TTz0PinHNiSM2SIV1jLcKMPxYafOxEi7tYfbDCZmYKshUtdNIvvyiZNlWicImvbsnwqNy"
            b"+mSs1uKB46F4dm5HK0djL8VGTI0cmWevhHd822OJmYO+gdh5eoU7ZIUxhmpZMletTtJIncutobk/xksViKvB/YCsMH64jKH+JM4v"
            b"SZyLwBQTmobUpUwL3BDXD5K6kIx9CUEieI5ippICI3DxNyaJKUjJrKLwYI70y3uVNwS/Hsu9Pmx7Xm3HXV2i54Zgesd015rZOgQX"
            b"BSjGyg/rEGtTflgtHbZr04NRU2QRj6nFGUhLCmh3UUsDgKJdX6Jp17aTrxRV+rUcVUCEulvb/D0yUjdF9vTYQT96OPi/qV5+d+R/"
            b"gDhVw/vj6OiMAyubEVO9zzpUqtuR1Yy/G/dzqB/+NSxf2rCUCGNkBReShaBsQHFBTTXS1oLyiAUXq0zsuOFeg00uOkgZG0tKgpPs"
            b"mPuQy+4XC048LTjZwEsnfbKqj1j9LpbqZ2WBEdEjOk8ducbaWbCozRCkeOPd1bO5UrZKMSs7trE+uAdxzYcoQNNe4Ks/Wlu15g9x"
            b"X7VOOA+ihXel50kl2YPW28zRqiyfVYp9kgtL7pnfjr25zqYtq1p3rdz9eHYzNr+tJzTfnSH1NAv2pZWlJCSWlLFKDI4zCS4EK7Bi"
            b"msnAEuqexWWw4wRJlxbDyU+Bi4C4pd5rQdBHWbAvyVyPyLHqPit/ZOU9kC28itU/QtHVhrFqzb42IBKXezBp/MF6AUC+QPwJI3su"
            b"7U1vL0Hzr3med9xYWF45/qSkGvX0Xrc3U3DmbOVDHsc0Q9vaEraMXwRdS8cTT9wchtHTJuL3TMlS2VZDh8ih+jvOWx5D7GW1Upc1"
            b"KIn3Dx9A7B+u3r9vin2grfKZ5DvA10tgGBNjtVNWOicUj6t5cJhSGrRRzhmvhEAegQ2BUON4ICxxRRjSSH3NxYx9D/znELV5QwPx"
            b"oChQQxs7TSjIzgYYrSgfsQbnpWkXLDZpgYNUAD6Ej3zXcKWnSC74ruF1nt4Bbh6aqR1Pp5xyieYg2MHvkR6S9IWm5XF4zobI3ImZ"
            b"Zge8ZcuZpZ9iNGNvz6+EpbOLSZWwJnkP7zolDKNOGfEurvSmkfiJaw1UVS1ccK2552cz87iBwaO2M/t9UKazKVLNWJ3h/5cV3wXJ"
            b"w4fo048oC0hpLXNBu/j1VpR7YwLmAgWrnGUBm/g7lfRApaAmIBsMItZxzlFSp+HqV/QNel+tpSmwyVwtLYEOELUs9kutvCnoHUGd"
            b"OWTQFOh0AvnWFCAt/ejIJcgx3657kA+SDqaYqmGefQth3GVogavpEFlWuO4b3KWV3sbiL9yoyouXVIh7UnGeSCMjSEk1oaKkKpZn"
            b"7jgV3H788IIP74fK3BtM0gZplBUBiz/IqKR9fTNBCa69RMYSwZwl4K3lKlashlrEpMMxn8USViAENEiMYmmiENNIYwucmm/H1V/4"
            b"LecirVOuvnxEwR9yMa8x5ufNhH6A1C/Lh6vbfmy1V2kd02g4OMKVUMBgrDOR30b9mCt99qug7my7HbjYJlJI4KamLbJTPdY1hy3D"
            b"JbwqAoiK5TrCx+JM30cDIn4ZR6gDET9jcyhW7Yeki4u/5bVQa2kAvaUj+6CHOunVjv0Wznu1ZNiRhQeP3svS34pWAvv3tXxBxf4d"
            b"/bqrQsU9Wf+cQ3lW9N4WQkgjVK2Ui/UjUwzF5Bo/wYhTbYAEq5kTEsVPGKNC2+Rz47xWTMjgFKfvkGclU4XWz0ocXu+9XtS5WpNq"
            b"TjgjCcCWa1pfihs6qgOuKa5IT0fDrTETdMvmfW+3YqYub4XPQQM1MZXtsQ0J4SguVeFQ951aisGh+70wTkdt7U3O8bhjepWlwb0z"
            b"Tqv3L0ZSXwfC6c2u7l1T8xnVFE+ppl+RLZTjhsBQRPaqcfl3FpH9Ym/3bVRTBSFITSAoRDHWJGghgQsPOiBpAStrCQ+GexuY0A7F"
            b"kyNeuyAUAy3xzwDe41Pg1iPU1j0k1hdgVkPsl9x8wtCRHdkHs6IOGP/KTPGWIDnCTbXTPDIa5Q2pXd1FyQpPqSOAWibVQXIqnVSK"
            b"igmN17pXPZQtPR1/bISr3Js4VOU54v3WtyvitEpkVR4bT8Kea7Luxt57flynXfhhUfpdcFGlqdmR8N+OfjognpQkSklvjAHFwVKw"
            b"0jDHlAkSBe+1BhwEdhI8Ip4GYEFZ4RxGiAcL6h0+XveImneJ71cmIyOQ/UiJ6DLz8zqfcz4rO4Bdr1Ak73jWLopTdKPZb6hNfmD7"
            b"FF2QvGTiMKblHwUMs+tYnnwRwLyffM08A0pwIkuuvPfqXjHeKb3fsgOLK/XWXmYiQxWvYv47Ji0q1jhIDoWuSlimh1FaOU6uJLHh"
            b"q5TYckrNHOumbtWtZuwzIie4FstuNhZ57M0apt5q3G5r49sddXCw9aq9u+CgobLyp34Zh/QoqdJoBH6eXTroRCgt4nc8fiMw0UYH"
            b"B/E/QscaV5FAFLPEKSKc9LHAFRY7TJSXwoRYBjsHxPxpq5yTDK5SGu5wAq7wSucAiwZeO0JATH0GFicGBHjgXd7aPQ60V2sC6qGV"
            b"Ew+ZcuwmFlul2CEoeBGMxYtg7K680hgZdK6b+U0SAnK/vUDjwjDwxSzvIO+EgVVkiukLZR3ZHPqsXAAmNxP/yNxLTmUA5NTa6+xo"
            b"uwH6rojVmqL3v/tTWnkDcUEZLomNBTw4j1QQCBnKwUFM14owy7310gVDKWZWMCE5tYg7IxHm8f/q5xJir2iJXJJcPazyB5zV4Sqf"
            b"HkG9o1X+sXV7qU8NK62W4gPui7f9jxRBF+OcNWxPZ60KQAqLP6Ad1qnRWohmJM5DPm2Wfk1hz7Ua8D0O7Hi8tTxqoVVN1NFO/BzO"
            b"9YAD+1Co+o8DW4ELlGMIucARJRYz5TFmUgtNKaeIc1DOKooQiXdJyaVTXgctAWkmIQhA6EPmg3T9XM/wAueOrqf2gqIFK9xS2Bdb"
            b"h7N2Ndl4VGQ+3aK98HWG15BlCERXwsOxw5FfC3JTYEE2bf5UO0TwgUPt0Kl1YC8oJppVQ3vBodUqLPaCMDjV3vYlvSHFd0rVfafV"
            b"W+qkY39BOfUXPEM9zfX5q4V2p0Ta/vbbWKqKhQN9OhqS+zLisXHqS/0FNXIOCek4kUJrQ31c9FLulRWeJz4/DiQJitKkA60M4nGp"
            b"LD2RPq6GmWfhlxmXiNeinPiJ9XVWfycPhE02wBIdTaDXXCkryfpjojwZz6docfSfyn59UjUdwkECvmDIuA/JN2eozVbk+Nq1Yku5"
            b"RL21S79DOsvFGIoBbIwqub/M0qLs32WOzurPuwYenyBaByJ4N1PpE7T8OZE+z87xLr9Uq+rjJu/uxPzc3VzZVM3GBRr+Nndne55l"
            b"ryBXfcK6+hoy6qZx9ZuQUZpQYnTACAsMNiDCNcKSatCYB+kRIGyc4iYgIxUhWFviLPMUax3XvOj3MfMnbE1x193pxBVVXKC0iis0"
            b"2k6xtLQWl6mZqPWh2p5ZQvynqPXOsaM9W3PZnMBRXqZ2i+R+jSqWZXK8LaGG1FT+zKKfr6H0AzcfBdHe3fLo+Tle/m0PkoKmlxWy"
            b"XlayI7Lqz8nKgwQXDHwNrd8EQ+ejHraXkLOH3wdd/yN4+poGg7AX4IxmPFWUzCuttXfOGM6tIQaQM9ISGzih1DrMnIzpTcbFstC/"
            b"cdJ+Z7pxdYR+1YjpzqT9UgPxKmxgHYKnn/zxEKSE5dE5Ia0ZlJhgSEtkZv6orbhsZxOHq5YD85XNssRXLKRKYJnNA6wgJFWp8x0H"
            b"X+kqlnBGCaY7N0xNdihhNMeDWOPLLar7k5aI8lf/ynj9tkbU0FianRpLn2lI8doPih/NoWCD68NJkVqJjzZZu4zgrwH4/6brb5qu"
            b"a8cN0UHHqjbE76rjHgthZBKSNFZYUFxpZZCMJ2ks1Rg4jamPxfpXBR/4G6rZGhH00p7CSUPh5vL/Qjkrl2qWrwGbslI7/0gRQqIG"
            b"GNTXijIL30mAJopVb2p5TSgi0aoGWakjVB1yRALO9c2Dg9Ddyorl0CwnSD/uRrwExbyNWhDYaAJdwlayFR1H5afzvfbJ1mg+9VdY"
            b"QkMlYVo/llM5UzmVMz3tjVaT6p+4Yv/gunzLXtJajSyNiznkmEucDWNcLFm5VzGVxbU3cYEYz4HEkhZJzbGiAaQSsQK2H+2LHsie"
            b"j+Y3D2lHozbnUz7Kk9HIsDC9aAmNq2ZgXz/v2Ju1fTnzaB5VqCcWycfWgKgIlkfNqwVnuuqY7HRMPEKakoW+SfZ7w4N3Wk47Xz3e"
            b"COzhVeT6WGLzApQFpSg53BWO7zhfwxLMCSF4pSrxfYg+cZMt4entxDWebBmpuBZibfZJkfmKbXXt3YFV9hp5EXB0RmbaEES8hhA1"
            b"zw1L5RoK2j6asPu73F3DRz/BSf16Hl8mXw8q2hue0u/K/dcr2gAUaCKjKmsC1cAE1oFyZZgMnFotwFpsuKcu/tdQcD4oqq2BWOHi"
            b"pwhSi6TI68UJn4OLPstsvXDvaKrjsThy73TSvImbHZQBZ2Ao+nR5uHebBbuOxtIeIFG1B+v8Ak8QEh0StQ/PYRTGEjDH6BRVTiYp"
            b"D9L+JkGGAoU0X8akPMgOrLROmrCMr57jsOY/xB1lqzECFE9JAXiKAF2PtjECqo0ZWXXrUjS/3aZqeLdH+e6d5ROQ6FcECt8EEjXg"
            b"MOJEoJiFGeKCegacOGRDrNpBW8+Q0JgwRQjTwXqOGZVIcxqMdR590MsaPt2svqoasOTh8wbv0+D+We97hJuf9YsvyRJ0mrAzRdii"
            b"ysp7zNmEEZXOrdDC6gYymRS1JSrfGWB9U7PkW0IyyoASjga2AgfgQ46rTAXkft+kr/UVuOfgN2l3rMIsLY517RSTKvWS6waovH84"
            b"EnHpZV++Tzf4vnDA88D8twkHGK6JYwa85E45xQO1oOMnFHlGEEskWmwQJBfU+CQoG+K3wlAjCHNJ1PvHTf0ujt/uEIyuUHYnKqxT"
            b"+s716d9DwdZjD3XqS3qAVHRkrEp3s+vKkr2AJosXlvhPTmZmhY+VI49yt6PGcIlKPysK1yQalsuQfvJmHDm7HOW4Jf6WPmySewXB"
            b"z/25SsQiJfvMzC99Dm/qcfG64K232Imcy2LPCrU7K1zyvyaDMd5uGjPyT9iNZH6VZcxPHfcZHnO50VhTJRlYFKRWzoOT8Q4gtVAB"
            b"i/hBtVQ6gpCm2lKEsYrVOEUuhv8r4sOuoYgfySjCmYziAUZ7kNaiJyjfwRGWOrNSvRqGsRyySQVumLBB6ECJttZywA/EaCuWQXcr"
            b"SEEFm4ZbZtaYmJBONkfG+pWu11dMzCDGVAs48Ccy04KSw+EOGjbxJVPcNvm7p2qY/mx3Bn/koPbyiBQxY2rJqUhiM/jrH1YN5W8j"
            b"YthQImAR6z0f/T0QK3wpJcLImLu8YhSynjUDH8tVh5nFGryxEmvDjGQcvBdcKYTBUCqcUZgaCL+cEvHUFHA0TjwO9oYQ/y/40chW"
            b"0utm43cykxuepKx6F/wo4HIE283cavprlCLSNBEz2gAqcK1a04nClJPAFHH5eMqXj1xi41stQ7767Y7ZE3yd7z1LiIi3DBFXG/dQ"
            b"aFlVZcug9dYZbJiX3i8lK9Ss2noMHT66Fla21k3lKfMZtiF/5IjXkSOMwYCcIhzH8lIqxRkymuv4rzQAhMWzQdayEAjD4GnwAFwB"
            b"ZSgEgxV+nv1f/p5f8Zw5N5y5yfQ/DrjyKuG871pmZWzQdD0aVNfnvFn5LXgGaMG5azl4VIfZatfGemV8Dhctdi5KHFT6r7x55e2P"
            b"JQdSrkvfAUFLpRuDe/N1oAqp2n6hMQEf6Kqn8PIKMXFiuifa5R4xPz2Sl36dpspdWxhJpeRI3rL2KuQy3lDN+F6M8h3d0G7l2IwF"
            b"hrUNUG3Jkx5B4+61NQWah0P7rxXw+8ucYf6d/8uSdC0GxYXyQXiFhOBexH8MDoZTpBEiXgXLsZbS+7jsF8gHK4wiQsY1vfH2xyhk"
            b"kdMsjh44h8HIOeygkFW/xlfS/RByNZ/H9fZjF0AUc8rKwCDthvhij48g6zmWC9w7MjyAOvA5eZhXDWmxzQYP516q5YzCoJhUGOhG"
            b"G2t0ZmjZA1Oitv0WwS6yjhTUYGCnuhFcQUfeQEHgWNOgm1zihfC7Jd8rJt85LeM9gR9tvnvQ27nR97FKrhnGjwvml/RtWZ2/UZ+/"
            b"4Ur+rvFqTf5u+7PFwWvL4Gm5hPLrH7u08q14CIuF1/GjypDFMY1TSgyzDjhKwDXCLZZSME1i0YyUZ45KI4J2CnmpvFbmtaJZ96Zy"
            b"5yO5+/O46cjqvr7gsF4cQwa+AEg7qloNxbnIWJcnHUsI2tSoxyOK4hcrZBW4Xp5OfutUPrd/eY5KHF06IqJVzx1IzUKJTZAydKYC"
            b"lk61BD0JZrirL3suJjj2uH3GIGxlu8mNAdeggqvCd1NkkL904PWPxlprzqRaWWcYsyzWuBILFTxxAWEI2kuJZEw4lIFWgkIIRGlj"
            b"ueFAuSXcoLewmK+ILtBHReyDCrZT1X+YCK/XnAey11hBfMAupoei+aw4bV71QcEp+zr2KFVTM5D7ErCIzhQk2A7tqujH59CumRdj"
            b"rmIzrKuuLOVaW9JuAZEuaJm8NSpAoww71vUZRL6qVMUUCI43oXs9B0qWzi5Z3Q2HmyUDd5ul89M8jfuWcWl5Va1gDlXnYUmqRfWB"
            b"170FXqt480Ny3kS/v30Few1fcL9+fVPbN6ZgycBzyXBQllhrjQ3cBOViTpZUcWKCo+BxMm0gHrShVBHgzmEGjvxjPO8XewhPInbJ"
            b"KG9ehMOOobWH1ex1zOwEAHzJan18O6lX+lsdv31JxjbuZ+Di/m6VYzg0twKYdwPyRSjeuAX21WRWMmrO5wZCuYMQRpbXkFUDgQ57"
            b"COn4ZYcvJWaGyV04L6mMaUgFzR1rPrKpTQ77byuN6zK5E3tY+7oHXG8FEasjv0/ObYrcFZL5vqz7NkSvVcxIKQN33mjlrfTaY+G4"
            b"5kzFX0vOLUKYx48i8URT0IIS40ByIM56/8eiuMiiWPxtMEJdUmoNbtLzMv0jWyMvSr4BKaLnmO2GBGPqwtFKnNVuB4fGSHrXy22B"
            b"IjRwMuv6CGjJ3RUv+oSCx2Gl4D0iaOTzLXFPNh4W1MMtAFkpX9dvNa8aCTOTxm2+BvV8rWU287rlO6qAq4djw5uqG/Gp2vd/jVZh"
            b"LRVxDaU8Z45pbhymsQbWOKZiiOUv10ryIAmznljEvQAlmOAiya/FE6W/UIICFlJD+imPog20Fz8rH1Ja+R9WCgl9tCrpE3OFOmmw"
            b"7aZcNUJzD5GSCnKqtgzTHzodMGVnLBaxhhaT1USXQxbXsVYPXO3lLaWnskJjnfN8uHLjYqi8xfrIFHcnkoNKcyGp2Q+Vyft3uoTG"
            b"n1CZ4rIWaEyfU+woIeUaEgWMzgUp2uuTY2neSYr+rBal0sMuMbK8i2fVKJLv+y3dS1mp+lxjTcy8bM4oeG1tDRP+xFwBeJ/qFVLd"
            b"nwrFp1UorLWEGRe8s1IRTYKjzGLJlUDCO2DcSGWVVwRxygNR1scEhI1D8X6ipf9l8OQLTYSLukJkqSGa48kFBSxTlDrokavHTfOn"
            b"wNH3EMI5kO2BC4TmGMqWwhtobbm7gsm6U7iKpz52Vk50lPJrp8sJCOP1JHZNeDm6qjl0oV5zilo9oUx97KYNKUaiDpy89nmuTwnv"
            b"EeZIxdogTfoeW1mcJfZlxlhJvJFTzvOe1Fv2cxv5z1P1mpKrznRJ1V9KyKL0YEZp+b2daYcckcbELzQ3yHvOtZVOgfDWY4eYSmIT"
            b"UnhN4r8xaxtKqdGKUw2xnP+aqzDb/GS/AEsWj5DJD1PrCKd7CUExZyYfR4czM8ej/zBk0x+85NXEWVuC82K+CxYb3fhAH8aH8BGV"
            b"G28LmX6K2cNMur/WTeDcjPeNl/VGWp2QO2Tu49srl6BEJ1rywYWNHondM3wJHUJMylFjLU3YinyuBP17ZnQOS8zomu18F/F8021t"
            b"zNQ7c06bZfazo62cPLxpUtRe7/vv1jzeP6r3+GVg56oNA3U7HP4hDHqU9gNHGgMCw6iT3DqFEzclYHDcxlxvCLHYOcNljIyVOZNU"
            b"hZgUnU0Ow+5DJh2XWuD4DGnHRki75zRAv6Jhf8ECZCYROgDFFd/K2mAI1pQ43GHCFuyoggdgHD4Bxg3z5vGYqeLJ+LlGQ3Rr7nRH"
            b"zSi+5W6GeCPVuR776A4gUDk6pYT0e8Da5Wn2yMfO4c+quH2GPz2qqs+Y1ZXMZvMIBo9anvXPFW+bwu7eaRhyCrtziHolXQg4iOBU"
            b"MMEQ6TB1WOsQK2ZJNXbKOkd8rLc5ccZIaQIBbLzB8O0aG+iRoAQ7E5SQT1nR5ezMMD8sIg5tjYnLnLxKvz4MFtO9gTDR0vO69ohI"
            b"JQ5TvSNvW/Jn1PCo51IuqHzSBy+9bsHAUVhwGmLRsEfLuKi6p6SYdAqUFXs42M8j7XFcHuXAJQcziWg7uV7/3v17zZFpesYkF8c/"
            b"2lI0dbvESIYrEMhNDYub6vVzx/QzXXtc0bH3x2e69ruP+igJ71y/igv4Ea/PG8IWX6JZl+RMrglevK2r4YB45Qhoa1ywLniPDUZU"
            b"caWYoFxzR5U3lJG0SAuWmFgGE82tR4FozH+djM8ld+NqBCf/U4fT2PlsKT+vB5PrSqwq6+jtjM/WVvGGvlAT2Z1LSjoDdIZaR4/Q"
            b"d35r7jhtr+1lmSNenicIjq/GjxJG6axS7JPyPZTcrj6PypNnz8w1eh6o9zx49Kfdc027xzEc4iqcUI7Bc8/AGo4oY5oZT8AyrITE"
            b"ghrE443XI4Udj2v4+FULXhqM/xV1Az/CqT0Cqd1HqF1W8B17vp3aFncud4/LtQGs7AF945pt8eC4V4gtD7gg4ox5LFvu8ViJbe5b"
            b"jKuWBL3mWzwRjRzYFt+Dq/1PuhZ/E7rcG3Bq7ysZOfIUJRSacRooKA0YOy2JdrEitMbHOhIsQsxKTWgiyzGErQfFYsXI2J+F/Aut"
            b"4a6ax9cqEHTVgejL2rECw3Bx/QqbjIEj/E0D+RkrDw37n3NW3hEOPZJb51uxTw6+q53a+rudjDa1njX1Hjegzr3Vc8Pcu8lELEJp"
            b"7VaD/q1pyTU5ruMoLyEv8VP+k0brE7BgwlKMlDKcBR5MgGytHJOsxBgF7JyKC3slMFUgY13gdIyjnpIgKf8xKj0nxkRPGzj3Kj1H"
            b"xa+vqPQ8nnnN7Ji/otIzMtd4jUrPgEJx05BortKzD7baV7im0jM6szOVnrld0oD6/KfP8xl9nnfKTHypcn6TOI+T8SNFAkbMGQtC"
            b"KKKTsgRXyuvgCOJSUCUdochYbxiJ3wIbkInlBXXMi99Ms7vGZrvU6bjKfL5E2bvNjHvADD6QzFhuE18lx73Bhqnr6sLE5JTL6k5x"
            b"oEmzq7X527h27B78a8xxxlOOM55ynNcoqKZkUHvFjYBgfP/l7hQH1aM/bt1LuXVO6WAcTlxmFRdbVCebCpLawsLGr6APSBFlBaFB"
            b"BgwEG2t5fEIazgEh9+Msi+6YAV1z9nlOIGjik7R6MRPMr9ScF8vtzYQDVzYcR6V3thTHF6GsJ2ppdOh/dF0s7oG8W+/oTNZmOqt7"
            b"LrTypibjzwJa9mAECk9PNAjhM422sstFkHC6Thvy9xnfIi4wkjJ+QO6aOo8JzonQUlSAFgGgamvlPNd182hrpD8/lgKa0aTP4iuE"
            b"2UNww5+V0Zsgvk4LoTWRDDvKpIjVeLwPWGyYBIXAIqcdp1yJwKnwNBbwLm5YpdPdgVH1BohvnYleCkGbIyHut4uPTexja3TQoe5f"
            b"rG4RLIDUrQwlzzTXM9A1rwDinb4E5QxPV9obOrY0UiRdkGpLt2GDcc3LZbTiIHiT1ONh0iIg3n2bxv1DW9NBhz8fJN/En/YvepWV"
            b"50wVaIbwWmltuEKF4coY9Kjzc2du932RXh/Ec635y2BpjMWMWgrcOq898lZg5S2z1oPl2gdig0HAjNMaEHDBmZVSxTKRvQHDNR/S"
            b"vVpJ4iJO9uq4agyauiiqMx/yncg0jPsLYjh0G3BvZ8TbQ+Llsws4kj/bBm6X1/WVXMUFk+MhYO3E5HgMsPvq2O0GOgxPU+PaTCBV"
            b"aiRNmjy6HJ89U7UHuk5Bzw87NhK+zZCtQY3NegYvGrO9r2dgnDIGnDEuYC2RiYmVIxMSr4uD9sa6IEKsGhkiCjkZS0EZDKWcYqbn"
            b"DkRnY7ZSL74B/nBSML6IsEAWQoDcnXB7JHt+dqMDwDHiKp1h0BAYkhTkNqtCA5e2LpiVLBw/BqwSEMJbaM9tUCUw3kDZZpmJRwc+"
            b"KCusLZ22D8yy85sqy2ax6gp31kxH7hsaWA9RliOerBnzzfLOAnzcF5X/FSn1nY4FDTXrLmlrnV/VzIGdJVDS4nevF5c0162dL1aM"
            b"zw+ljoMoS6UEo4JEkiHEDSAssdFOIsuJk5IahYPEAQtHA4qLYYSwk7FUiCVlUD+b1HqxTXm563jPH2JxPkgPKG9aakP5xDtcU1qq"
            b"stSE2dM4mxy4vH6OPFow9Fb0jxeqF3i7x/YsOhgls9XqIf5kjac6qlggG+AVbYBXQRudgWMrt8Rg9DyV9c6AaZbgzvQFFo5AtUUq"
            b"fQHSVogPq8XHNeQfl/V1XFYvqDMyzVuMM4HGZQ8PImZUzDmKHzwHzjPkiEVYSamVECJojoWlHqifG6f9DpGuC+njuu3vRV7sPZWq"
            b"a+JgFyWyZs6VvXBCPjfZydxWgpUDYsbSdoC9qzlS/sqelSnqmvJXjsq150opE/WZjJ2Hc3A59WQPTJuidWwhTBdtta+odKW50p08"
            b"PAfD4nVQv63IG93dKRgWrx7CUFsI71AAeRj4b8Cq5iH5j3S//dPreilK1sc8y6i1UkivvKbaGaFcANDSplGPo1RZIZQ0ihkwhIBB"
            b"AnnDtAmOP+qKDqFWLwJa/SMHywkUiB/qzMsWDwOy5zbm3/me3c0ClS6AwHy4UxEJwueF+lBbTGxcNCFli0dqj1cC4g+1yqQM47b+"
            b"7szvM3ck0psknZRZc5ikVJvT+vJXVxVktQiZNdHprFLsurh/xhHi5kxoTH49mxbNRRBnjc+j0uzYD+IHeD+0+KdKsO2ixwPqVu/r"
            b"Gv58/V9nRbLnQE8U9xZji4EKAYEb6xz21HAZM2IyzpHax8zoDEvDIMEsQTIuDYnT1n5osv0mMv9z4lVDmvxkjC2eVDUfzpA7Lv3A"
            b"hH39aC2zk64iq0zY22XwiA9Wuab3M3c8rN8u1rTZ6KZU3sDkZPiOD4efu6zTSpsKD6rQ7LKeXirWlVwciqDj+iGHfUFNZTFWv4f4"
            b"59DBkrYtVqzS+d4SqLdYDqtwSd0mKxCnzYm93Wqosp2S7A7/7z14Hnurf5NeaqP0XSqDeaH69qn8eXtABkRVXOtTLo1TznqOpfLM"
            b"cmtMkFgaQZRU8XfWaqMcRZ5YJgUYiTj/efDTq2J/r6Na3QC8nnkQ9KIJG1IVs3YoBaMj38Jzdt0EUinDHNH3coU07fKApNKR4YNe"
            b"cCJQJVdeijd53J13SwfdhYKKXYixa011BPnmS5EvcuW4Uy4KGzVXSmA+DUZA8KYhsRtSNKhbtIQyGiuU/U9U2sAHXm/6M5bIJ2Gn"
            b"5cNzvaswo2mtzyzZudnAq7AWr5W1mud4rbrFK6xASw3YFGH58GHXfShcrmt93z+E6ZsQpt5BsEkwFrS3XmAmhBXgQiy4ATvDjTaJ"
            b"ncBZLE4odU54I7gQgnChpXP/AwitS2Cpy13iy+inO5CqV2LD7sC+FiuEgrgitVfRksbFAXRFCl79pwO07qrHzuQJ1wlaPU1bE+24"
            b"HXGu+bUJElYPR/qxayPjM6pe/2v4LO8pFslvgXOkkabakyAcl0JR63iqnxkHLBh1zEoLWCsHNhAtPKjA1Y8rqsfCXOSeNpd83CN5"
            b"ndjBEFZxvVB/JLQ1rawnzYqD7Net4v6oZIbVXFqgHDRT3QhBA1ExNmyCbKLeq7AuVCLdbKTQnfjD6Ywwx8DWW8m+xxn5OF3Ycnqg"
            b"qFpOcr3Qj/fOL1j2fLLu/rHyjKeC4ZcS/l+p/a5S2wPD2jOc3TSDxhKCiTcAqplmQYChlDAwQimXOMA83g9AsiCVtkLI8D8Bbfsk"
            b"nmvEtT3aHxyJtmL3MmBHlumWJwWuwbe08qukz1lGDDAee1dGPHFfHLRuTpwdxn2bWfIXuMQ9i2m7p5rwHKjtnDYx0mAYg9rGLY9N"
            b"dmx7+Adre13fOiAGzGpAHAfMKFUCBWXBO8Zdak8TIZD0TmBlFUuNDSkRxV5IblRgH2SWPV2M4wfFOLve4Z4wyrZEmjPaUVjhqqjN"
            b"kmxbZYO7Cjoncowz7YVlNSG3t8Dny4l2rigrB7JZrc62folYwRayWnsNPGzS+1NUrs2HXcIAr1XqUYJ4U5XklU0xWSwoq/Te7phe"
            b"peyD6QbPqM5vcJ9Jb6MEPw0zptmr7l7bY+ILX1rmVWsaN67wozb10hLhNaGDDzwb9tbG0Z9hIAf5nfLyfUGbefPjUWZ+W/MjAKbB"
            b"2ljQEskkx5whhLSWmJpY+AbgVqOYsonAKJbCljmgwhMWy1xKsFD/SvicPoKtPcCsPaCB3RM+7/Bhg/4KPWC65s0VeQALv0T4vD/s"
            b"ZeHzR6rm/YEnvN6tDwCHtjHLJ4p2dN8ITJefJ/nQnepuf9BT1d2jBP2xu/wMFO7e6E9ONc/Pn2nNx2oJmpke71hxd8MPHx5+VETm"
            b"BkpuYhV5rbtQsHP0MnbubVjiAIoJjA2Jda1HGtsggcRXB2y9DABOpt8l33VnDAhvAsdArXFCimDUR6nA7xHcfbZOvtRqqHRk1EiS"
            b"9yt6u2MWyTN6uyd17+2y+kzofCq8m1spsmry7nyO7tzpxhosZWu5skuRfJRfTAfNoY0LEFoRgVO9xnzH4KTYq9UK9BNNuhSZ7zBP"
            b"18YQixdxaypIS5m/1rfdpizEjqrepU3Bi4s33a67S2uCMyW13li7VWmfDzXRl9y9CfA2Gb1g8H50M+MLRfPb9HcDM0JJRJHBwjod"
            b"hOGCJxV1IwkO1iUBMKyBIBofC+aR5MjHKlsYQ7XQ/3uUkCpPTrLkQxLET2WViM2KXgDq1GuX6M5UPcWVm9Q5vyQ/ne6LgGB/C0OS"
            b"NoUSBEhUePESytqbQw75CinkGave8TRuBsMgVRxp9pnDMLo+w01YxR8n5MgJCYFRrwTXCqnAgHPmCaZMxrIVEiyNaYZj8mM2Fq4+"
            b"OBRrWcMRUAIgYha8nwTZtI69h0PbcqCY85bxI97yffxaVZ6O16G52mLr25+sVVsm2IYc6CxlFyaYqEFbaxvihKu39UoHwOGrSLhB"
            b"Vq3RwuXI/aUZZNUeYnxRuacr0IdcvWWliNqGawnZtG7eizB7JumNqXBnSW+Xuhm2WX+M2lcrd7Pmw4uosdemPCODx9Q74jTXTFke"
            b"ABnJKOWeWuUJCyh4w5W2FHvsmbBacB+f4UH+OY3/OY3/OY2/Xm125jR+xiyeOY2f6dDuHjgrPottmK2a+VZR5P6cxl/dQw1xUe2J"
            b"cUE6rzU2SAqrhDJaeoKwRpLFepRbB4E4pqkOQXJOHRgXoxSzf+Y3l6Zd11xtrtvDXHcwu2S6U0N6N0DvOUJgRNGYEvBuuPUMhW9P"
            b"4LlPuPsUOnTukhJGlteQrSXZxBOo7PCkGQ4ld0W/x9LeMzMcNhV9ZNNh10jMYf7bPSl/N87ab7DDCQiMJrzwIUjQWDlD44fDB+m9"
            b"AG8khUAxU5oKSyx2hnHMlYekcQvM/Ahp29d5+n4LadsDw+1M2rYLPpO25QO5trm0LT/lMOORHNpU2lY8EmE7uskfpW1vGu1icovm"
            b"O3f8GivbPoeR3Wm7u3zYTvX9hLLt1/1zv7DUf6WybcDgUMxiTHgNUrD4alxR77HGIDwi2ksWy8qAPMPOKqSFSebkXDFBNWj5p3hz"
            b"T8vmojDOexRvrorSzAR6h2zbt8rofFHnJp0+5gTTHRrVv2B/iXJ0rkOfXdNnHjq+J7o417vBp3o3+IHeDT7Vu8E7VOpAIoDt+V5K"
            b"4U/v5uW8gYAZDpYAC1bF7MsREg5Aa6SQyT1Wxb2SzoMwmIeYsS0ywsb7umQxRwfyr9Cp+NHS/tG6Xp1UfFV91laGV5fYLNs+M9IM"
            b"aFYPvyeAselIa2KUmLQTnFF/MEflBbMUexm5DUWP4YIvt4cjRhTd7PCWY+TvGiJoH7Yt+RIeQnmPQ7n0guVomJJCP6ArUmo5anMC"
            b"KSb3LyjGa/BYkSZH5Ev/rK34HQuGTeUW1yq3FwhYvOqfthvTZupdhduWsPXt6bBvWMO/rZ8ab+3GoxAwcTrEbzxhhGhlIeZXRLWS"
            b"XiIqgjYcew6WBkFBxGU/+CRHTuTPJrpeRYTeIn7mJCwJki3WamfirUVvCilf7zzxaMVd145et88ifr2OngZtz/pk8nFTqsUUYF9d"
            b"t21SSm6KrC+HSz/lmkJLjX165HzadZH74OTLLS3+EK26Gu0s2cRyJpgBbGYPcnhGzZgqReeSlnHM6je7FNGtRwTePCKewpwWpYqb"
            b"djgT+tTCtKpZVw8NcebkgSPsfypLkDXAXqVE80eS7YpdgXB8OZM0xjELMbdaK0CClkgbRaVwwRsSa+AgqZcqSKl9zCCSMaWY5P9T"
            b"xe6pNs1zg6dr9Fm6NwGgVNBVa3mFjXbFdgnFRO0AUEx2jGlX7aaolLEfFLuXStM3VMTft9i9m2Hn1e6pBti02pXTarcmvO7SLmMS"
            b"1pEG+1ftvrLalcQIisGSoIkjwnFklNFaJB0C621AwjpiSZoQOKktUZRbRWM2Jgwr/44k+0Vs/RmTlDyYHtXpFV8U1xJHMNSqf7Uk"
            b"2C4jDsj66buWo0HJBihaxbdnVAKzDpbkHSq+HhPs4VyWdB7/bDusdIXRtzVkCcl3ANZk6R1M3x3+okbZJP3ipnuSXjC+fHzA+CH9"
            b"Hl+7xNGkgqukbFRw1S5e0DlM7NAHVE6Wb+Y8RxlchZZjc0aE3M+db2yA7vDxDZbQRr/rLnGAUhZL8nuYAw6FA7X3cetN9l/L0mq3"
            b"2Bq7FtbdJlttfrZcDxVMbG33tkLoSxO6AX/tv+ewHjYPPI8bC4vr29NuG9+eLbNv9wJa3wtofS+AmrgAHyHkXlX8CtgYwplyCFhQ"
            b"XmnlGPHBxgfIx6I73gSEAaYEcch4LgAsaOIVj5lGBuq+OavrhJP7vJBkkaQRKwDroEiza76ICqQF1+R9J3U6Xcd7BKMFB7Au2Xfa"
            b"Vte7yWm4JEux01d7nQWSBRFghSoMim6+3EBL27q9ZfeEAshhcu3riKbm704wvoMUGG8rKwQCtZ+B7baTPh3xH9jbLn3HBBr21g1D"
            b"inQ7u8thaEEJe+088kNbOQxQcRig4jBA1eyA6hm4IBXzA3BdSwYcIRse47deR2IIBFFlNXLSK6d8rIYVZ44hqQQykmrjgguKI2Z5"
            b"yoSEecplAEkTc5UQeAaztRYbZ+St5/oSn05425q7BsV+T9HcK1q4AyXckQzuY5EDWPQZyS7MUi0uznyISVVlH1NjOuKm0VjLvizQ"
            b"XTa+RaTQ4mLMseLVXusurfdxr5CAlv2Syc+2986YxS1jtkQ93bT4RdK1P1im9ku9ivfIDATCAnfM+7g8RdxDoEIrZgh2KkHPNLXU"
            b"S9Axl1NKJKc6mLgDwsYaG2tT+HP7eaXbzztU098jMP6E/vcTKuMlsnR/Nt5Do/M10DJPgfkvwjoHiqm9fAnM7REAqNoj2/vAI7WD"
            b"Ep3Mf7YOSSNIPhZAL/HPmv+kz9yd+Z6sWLuyyb5jtoT8TzZ1NumeqaeAw+nexos4EUE4RFat6YfCuH9i5G9qTRAewCQAnCPa84BS"
            b"A8IIr2VM/PH3SBsZmDEaDED8KOfRYazmkXUyWQR9CI38EsrxHKSsbmqGH6THj9Vo2a860BK5u6dfsW1vcAIpQC1HrI8HVY7jp8iS"
            b"cokaAdlCViCsOq6sXdiftLdPByzVeLwFNmiPh6ZFA1hIPki+k66kipuOmPfIZrjRAN+ZvQtgrNYA354p69Pdvmf3zhxzKhou2fjh"
            b"0ov4iQzfD/J41zymAqbgjMae4GRgKYxPmF6qvZUhFrrauaCZDNQGThXDgmCFjSIeUUTZb+eL9QSqIV8MP+SL4Yt8Mfzv+WL4m/LF"
            b"eumEIV9snYXecEm4k+LOuLFj2hib4r/OUtxumL4LHFznzX6INrbmtGoghZYP8QVfMbF08gdp7uXMMeIkGE6CMwKUQToWYk4xjo0R"
            b"wXOKQ1y+g3ZIaMtJInULzWIUEGrjIoy/Q59gNkf6wtL+fF1/YjhAxv4wVZnE2xFSPC8gFcufz0wdl+9xLQkwBX9eU0poXQLqimv3"
            b"COh0/tP0FyGyx5PWsLxZdKbADDzAyyq13qVnWZSwmKfpOY7qGo4NZQWsGPDcsCgvy+9JXs0WtbharuJm6bpisniD0OL78pfvnK12"
            b"a4Oq8sFGu7b9GcOjEcTqg9T/JbVRpDANDBmuQ/AScRAUaRmLsyB1rEQk19wSx6kmliNsCUvhAaSysawzvwKh+lWzgEttxTs9RfUG"
            b"s4D+sBdgpFeRuxkQS5PFNun4Z/s76obui741eUwAy88vNoZIkf10a+3/rrSDJRZzTHgHdjpCX3PQ02ysG9XfTOT0/JnxROf8GbLU"
            b"fPujXdSqfdSufv9wqS/EpVKgPFCFPNVCcCET3ZUGMAZ7I4whLi5BMMPegZBUEEACHNNK+KBiiPqbyp9O5S+MjG4oVD0eh18e249t"
            b"w6fWsTmsCBCgej1Nlj/IaNqf17Zkuzp7/TyYR+XjFhdvtF6qYrIy7gbcnOZn/EsST8SMVteP1KIDBxtDupxMfBMEVyYEZ+8jh6Z3"
            b"/WSmJkgJzO6pX50o/LP/1jlKNZThsKNQT5wF2KmzAKtrW7brFnS/rywExhjVv8n9Syf3FFhCMxNrgHDFLZKeK1BWGQDwnHgjhZOG"
            b"SU+dxi6mi+ACJSC1dCqg36xR+E/kBy8t91+uUXhDRHDibnDU4eayKtwvqA1O7LjGA/oSUvL/YrJ14ikwtNmSu/POJZ+t92hurW6G"
            b"pBIjJBU8dSRTeC6kPW7FnvgK8NFM/aqT4Z8Y4RNihJRJEZiIt/34f2scQtjjQKkJyZsFMFbeW/BJltsRHH8vkPA6WIxUcIT9cCGD"
            b"67ID79A8WL2pCSXbrGXaJy26BykUY4AdH8/2HNaecI7KKgZsh+cv5QHphAlSRPpZMaMWfgC7LXRwQ4EgN1LVHlUWV31cEX6FRUl7"
            b"422VSUYTm0LK24jpUrCWVtdmVVHsa5907uYkXt0bGjEpfdRKWt1m/iBWUlrt1pQBgJcKuUEtVfoyWQBosLXLb/VCXLW2zJ9KwetU"
            b"CqiLVQKyFuKLApUYO4ipFUuGuWIm5lqKtRReOGy8C8lDllJundAQy1tvr/WAYWrNLSeOVeeGsEUJb8KqZS8C5u9eJdMStSRFNvL9"
            b"O6yjt4blNZTswVr1gJDCh+EbW1D7tJEQJIckU2kZ7oCm2XupxQx5E7v9mQYGCUtbYPW1IRtS7BCdTjdHUkXYrj+wIBBme+XgdE7r"
            b"vGsfoG3juY5qtZklNL6wOfzuoOyem0Fe3XOop14cakJrpXpYb7EHyZmdJueme1CpaVcPq4AdIPBY9/BCNSv/rzM0IuT/aqPnsr2R"
            b"VnFPWmVXSKs14nNGWl2mCjv4cyF+9Kk2+3y8OdESEbD0PHiKQQXmrUGBp7ZwUt6KqZUIGYQFF7MvVibJblONrEZECfI1pYKlu/u1"
            b"dCseZdx+vPKypHs1Rd5QObiG1J9KxA5x6oPZH3nsFb7xl4ZX8fROMujGzNQbtux+3KfwC7KG7NoCObILBua7OTjzzzDajA9Ui8lf"
            b"1gtNTb/cG6DiJWyGNbP3lcPLWzpQDVYk7vj1ytmVnVo5g5uZ/xafFi98AKj4ADtQVVbC4O3jmXz4/JldJHzXXVxbF/WjrXOBdwrB"
            b"SKHxl6X9h1oF/+CGMBoXOi2QFUZq5A0VnjnEhTMMm/i7oLHhiBOuFeMyBGKtBqmY51wCA43cN5SxuZD+7yjdTJsc/CnO2etvFVeT"
            b"/yPyV/fqD5lc/CBVcIXJ1V+1DzG50lVcqFx0kSvv1Mb6F8lhNMeDOADljum+xJW//dfUa9Jn47ZrGa9ty3gFyshFPYdd2xzvawCo"
            b"bgL1Y1ZUzaEWNYemuu/laqqCfjY6PIjYfNBO50+oZpL8DSJBc02oJgLRWPVDwAyI0CoQFjM/xcoiZIF6D3ERwYRHziOqlKBM/Bis"
            b"yBuMyA9YkaPR+JV1yFcgIKcNogfybdMJZX8XuCrHkBXfC2BQ7W+nsYFQ12GDx3tLGTxm7B6BbX1RU0aG95ci+wCyGkSuQJTxDuNh"
            b"JP9vvBIZTiJv+5ynhcKdkSRZtSMrKbJW0feY0uXSsmk6NhVAGnagSLdx1MjZdCj5SPaX1xK/P1uDrGXzlhROb6Twd4FCgvTIcu+p"
            b"4lIKTYwWDAx3yBDJHcGSaoYYcoUcpzXRVAbiAvjgvHgGFMJOxpZz5tvTwg/4ASaEPRASK6fbvP4tUsPMHb33zs3u6KTW+poapFO0"
            b"NlSo2NDBYm0m9ZV/S3Wpr/DacM+HoVQuSDaxNE1IIVweXz5FJiAyVXj/0q229mvehce3iRl8r5Yzw3M5M7ZYcAgE+xexUd+so1PU"
            b"ep9QBXTSoBRH71PhJfjZSeZt+jCp6MOkog/jCh6CG/rwWGOHTbXQxujqsVNlbyT8bWaW97Ehcy7eo6nl27AhYASzmnrFHMIIHBck"
            b"GM+URpoZhrHSXvFALGJCpzMjElODAjYCaYPkZ0vmSfd9LZlLm2xcL5905B+048+YxwtI4Zx6zMZhLVOXbIjdOdCELybqrcoxKhII"
            b"UMsp4qN+ejohknu/qm0x79HnBOmpJPJMn1L0csw9QXpL1vIQPCZIkxWWKYcnrBaadM3cuz2W/IUGlt+l39xA48rfj9zoHr+Qigze"
            b"UkpcTH4UhAwWsODCYMpcMDYVoYnLZzTEn5ZaLnmym/BCOgJBa/bjRog3urTPMUJGU0S+d2gJ5u2aGA/XxBenmGtV3EAu6mt6DvSg"
            b"F7Eex6ZACoovJ2QVK3ZuMt1Q492J3Bp+YrLp+BYuS9pdrXDBYZc6VxUoiZNDc6F5ZXjZtXezTDosTedV//e41+Hcsnxv3oWDYKQu"
            b"/Kvd62sWL1YJ/drQkErJkbw5N9zwIc3GYnLJm35wZZEpl02+dRNqHJ9stNDbrWYkWNFJGuPM5vez+OOU8W+e+C/miQxJcE5bHbQB"
            b"4QwRXiqNrQNgkiGGldGIek+IVzSW0sCNQZxgouMKlfGfjZ1+zu54iJk+wC9wJzV7f/R4QnK5KYg2vgXgw61igMfDVaYcVe9L2ieL"
            b"u4Q4wbwcbZFn72mG9KvRH4PzyUi/fC6YE2BrK7hyYB7B51NkOq1n2w9xVSlTl+S22+YQ6IdPgX5fRWHP7Y2Pafy1Bsd/aOoW5MeQ"
            b"plIRRWMaIAqoSsg+z3kIJlblKuZfpDwKzoJHCdoXS3TqhTGecKID/yBR8OPF/OVh17g53JfpvdDQSSG9KxV1ibG/UQzLdraxAxeS"
            b"B9uLRjhiEFFFAV8K1NW6ctwzGHZL6AwGk4+aLUNjgqtkjGiLyB5xwVP8wgUswhjVnumtz+wsDi2f5Zs+poTng6fuDqWIdypIdITM"
            b"zhe2RH8JmV0WizeIL7kNIiveoKyskWXVxGgfz4rzxZF+76pwGJgjt9A8GLREviscr24Jb2p/XRZeyuQLY7uHZfLbWscMW8Q8dTbE"
            b"WhcACYwskTgLu2kpjFUqtYuVDSR4SxzyWIPTwIjBStMvlsZPilN+RYGXnYnwymd0ZucDqYMexlytEp1a2FfjpTZyYk5fBljHzvdM"
            b"jlfeV0RKz5K9Q04z2y99Ew8N62XASHHDCpSHuWUOKe8Dcwy0pULOJO5TYDGNpoj24sWsp+KkmAwCiCVJ+6Ho4uJveQWhuCn2eyfz"
            b"zsSLlvZBt9U8PugD4+mA7wB+rprO549+ovbvcVBXeY1/ShV41IgAwyxGnhGlBcTE6gWlXDGlOfJOI0KotlJbHgQ2TAtGCRWOchSz"
            b"MXLwPL3wodbvs+JyJ+IYL9CB64/T+vtU0uMrVeI+qkJs5G4gq3nkrmiullXSTMsCr1Vvc5bpUDx1y/DSMVgE13YwcAeny5Hp00oP"
            b"Im1HdF4O2jXN2TjuwgjgIGl+0xKC3AQljE2H2VSzgk0th9lCrsbbMzc0zZfl/48UePugjNuSsjhQxBTl2EgKacYWP6s+pjHKRDDK"
            b"g3UMIAThkDM2FoVWaWtUPDVHpDX+Z/dOr3r29F4M9SnelZu4ptzwsnbocRy2orbo4U2koM17R+C6QKMrde+IWjvYSixZlT3K1+OS"
            b"T64J+1r/NkVlBQ+KSVdRHl1yBC5xT7dJ76XEZ/TMn0mjY75zxX2oHvbqmH/90Nf1Qzl3BISQ3MWXtNRxrxiTSGnMnEfWWSc4VQ4p"
            b"ZayMS3CnnMFxCU5xPEnKv91im3zJD0I85KqNcZ1Xl9HHlEtGqsLXV9vX2gFTI5023Z0v3Y+o00tr8mUlvHwgci0ze9vlVWU2MV61"
            b"hOE/VTHcxIDFKDZHHdU5r43W80Tlk6Gg8OEFhmLIObKc0L70Lh3Sm+KWN5gLKy0AGnH1Wsxy9fiFxu93oThQsk6hqq1CV6Ckql23"
            b"Dbk1QGv/38opvaEskNw7vSas9iG3igcZ+dqC/OBiAeMF+XuFiblEwnKnNGGOBaosN5hYJ7H2HsXK13EcC1zFlRceK6EtCRgjo4WT"
            b"IiDyJ0z8VWHiK2QzmmcLpLX05ZWrLzwlTnyRRHZZOPMBfWysXrncowiUG0aN3pqoWD7gtPUyd5k6lg+PGeOqESbeX2pwHXN0sh1h"
            b"HON9N94C5Zq/ZY7M7LbnRDCfIJ09Y7k+s6YkNceM/CjC2f+aBDG3VhGqY1J2gToXMCCHKRDCFfPIs7QQE54i5D2nVFujg1M2VtuC"
            b"M0Tkh7oSX+yw4mmHlQ3YWemTVX3E6nexpIKSG+MPcarkWNS9QQ7YY42GY8myBCm+Rm4DzyYdixIT19X0QHW96cGYDpGep7JUlLDp"
            b"Sx4Uzpo2LVoJaK2FWjxIOpgqQyxVA6geQtQ6BzWmisbE0kltsBzXZ/a3Bvbz4dCYyXU+HDodQk0flaHSP+8RPPBPO+sExL9RvqWO"
            b"+gGvb6a6IJIGujCISYOMc8rEJQ9GSeEGWQ2aMwWSmviBIkQwy5TUiBpAAmEOv6wBoJ5ZXl9uE8yW/wdWxaXl+oXsdGxM0tJJreBB"
            b"D9wx0NpL5QNZ9rHwDho7zvW2wAfjudwpEGjtFGy8h/FVWlzXVXYp4njUUx6edA5OL4Mpobi9KivCqpNUSHFL17buANxNplxgJCW/"
            b"M4mnZJWtWZwqqq0VNVoTCvat4vlW9xBqOkGRO4NKoX19pnKuOMBSVw7BLlfTbTykFHwP88o6+daiBks7YORKfsnU8l0dgRC0sNIA"
            b"0UiFEHO1R9Y7JQUj3gjgmjjjMfMBwGDJmSbeekwcEwKk+82Y1ccgU1gkxfFq4LMpeVHaLvqXSpTVFhOnAuZDNGqm2+cilMNeaq+5"
            b"pT1eCknduaperVhVh6qVbVUrXqm2Fe6/J2LhIrxBMd2iqys03StG52slluTIKodM3u3YXsJ0zyqK+Ix3OtA7tLAio6XLKMs7f0BF"
            b"y0dMoRvFkKy7DM8kn7tqEvYnuGHyvxWAWoNR1xQ71pSUJ7DUhSFWMRbarRWgKiuhsKYBUDG/5AZild9QQPi3QFcFUZxwG5hhXloq"
            b"OFFOEa8RjomaBQ9C+PikiStQgYIGFItq6rFy0mn1zYTCztoDr7L27DutfZt11NCFS3ZrB5/OfKcoIK29YytGsVdaxK1RMV7LZJgc"
            b"cqplcJgjrlQG/ticLr14jgSEYb2GuLmZNosUXL72SMgq+KCiJkrEc/grwAoSk/dGxjxxcZOn5W8RvW9t3Or26Lz87fyJq4p3PPP6"
            b"/o3Skg27/Pn2duihBSqooVpKb63VVnouAvYuNdBdTHXKBWwsiiUrpkxjDgCSEqmFUt5QaYT9aC/hgMB/gAu44FjxmMIqVpTRFUTX"
            b"YYg/0logX1qYL9l1ncxPfN4HgrgnXYG2N7ktw8dq6Af75OprchwcVdMsQhptWjlC7w6leTeilxrNv2bSvGhziD+c1aLOTnIfokEV"
            b"dOb1nTZ7DC2vwQmAaN49Hk3ZSli+UpviwU0gwj321bxlO3tm3rJ9tpn7mBfwKtuhT8AQyvf/ETDsRkfiXQCFyzIGAhjRAZBVnFIL"
            b"0lOAQAyNVa/QTEhnrPBSeBMIR8CYFyoE7YzR1Bvr/mQMbmkOTCZIrK/0LiGE+SoHsxs9oIoBSgaKB5fuKvl4Wc2AiCaB0uWNjdXK"
            b"L96Mppo3Q5fPfMfIxhmEUlm3iGcmm0s2T7oEu2uGam2LDj4bKTC/24+Zwi3iBY2SQSNk0OgYdDIGex+43SoiBlBrGMAXJQwW4Nof"
            b"ZPeFkF0B2mIds63lShrHjA4YS645iODjLwTziiBOmUScxURsFHaIafAQwMu3eB1PesEf8Y4bqAg8kCGQ7dgpIQtI/hahVQFgbAyx"
            b"TO9SE6EEn1isXesVt22E9aAN1/860V92vYGEr0JYdhT/gSvTJTmHfKgcEO8vaL3e60EblkcO+JJYYvZwu9V3lY1GIq6W/aRKYqRy"
            b"vDxTg5GnajCy8cQ8bO0OxN9OYetWn/Wz3dQ1uUnl4v+EMwIoJ1oEGhj1GrBG2HjBgIHxDIQh0isaLDIkUGcFD9ow+We18Ge18Ge1"
            b"8GWrhfR3uMkamzgtnJnnzJ0W2InTwoww1nllVg85VC47f04Lb8C+SqYCUckIzRplQRnlaZqFBcWZNNgyy4OjAMQpS4h3SVXUU8yE"
            b"FI47+YdQ+EMo/CEUPmJ2PEMosClCgZ0gFNgpQqE2PqutzKpMXSEUvrfF8W9BKEiFwQvEqXbBB8et5jx+UggLRicasA0KQRrTUaed"
            b"J1ZKkFhrEhjHArs3eOJMDHE+I6bQNFSH3dSMx997nmwS1pa6w+5to8Sa0+IwLDUgacrOolE/oHP+VU7QOaGX1oBoEzruc79EbULf"
            b"u5ptcA7KN52YqlFzNuMWbmqDpshNCYau/d5jx7u/VQ2DDnIw77WomZnKzJoIZ6IIs4nXqRrMXr3+UDuaD5rOrAnNeaUkxhRZQxw3"
            b"3lONijcj0sFjZSgnVuhYgOogrYnPxlxHqEIx1Un3OZMv+noQwjV6QjVZmc9VDlN7fhH4cBXSMEMqrIAnWhe1/VvIzxfrls1lRq7a"
            b"LqhjYzQ5vawEuqu7DcD2BT/f7gFTbEAJK6t+ULs8whAVUEIWNVeOtuC9fn64wpf/VXZqvTlyt8i/CQS440azdFMrfRey6w1M9F3k"
            b"8hhXDjbtM0dvG7nJAR4e8b6xusmwfj8FgjLif5AuX6RB8L4q0RORvAc4c5Zj43V8wI1UgcVC0QJ31mPMnXGGaE50DHAhpiHCY3Xp"
            b"yE8b68uHncihTN9Ollr7i1v6eqkYl3xspPMk6gCjpZcqV92+yvhGPCWZ9VoprKP0eL7Y6YQTj0vs8q3rSY8ktGIc3JAHm2lu3ebO"
            b"vkp5axkkVeovj5W3oMrGMFPemrdMq6KUQ/3wu/Nq29bpKrE68x//BOP2dJavGMRcKgU3NDiHiBIEaaRAC6+CYdpwqTyXjipLvbQ6"
            b"KMWBcMwZlgbDg+p12DZ9omkK30fiBYq2f+Xcd7SGSr8trVLaePzx3mCQ8jVLSmgDBz3TVoBguABOa3DGa5vbgfMqXzgLqgbwbut6"
            b"fizyFSn5HsFhh6MmV3rt7AaLkOyMG49jpSUs3vBIJYTd3Ljyc/EHrAG4M+BNTz1HFiB3/WNH2gJsKjz92D929MzUP7ZapH8fPsB0"
            b"bf4FRsA69EFVHsPV2Gg2PKrJrNtgSHEnpAfFDcT0BcHE7GYFJcwnpysXy8bkz+xjiFWYxpJSKxoIE9aIpDHwMwhR54P8B1P8TjLl"
            b"LiHq/oBfLjUfXwPUSl3q0mh8mRQlMeo5TnxtXTbHzVOYGSeKXXFlnMz7zzhRvVdLikozqx3Uup0COpxCZlClyAEpij0gRd0dqTPK"
            b"pUDq3sL7lBzFG3+TdusiOWrb7rYaQNMyoxlu7ZHff6h+ZhP71tH5YVyuYkGHA+ckaOa0oN5KT31cdlhrldJaSuYoNZY6gjXExOgF"
            b"pyQ9oAEA/W+Pyy/CFXkRJxWkSt7LSbdDDlFyaluzrQycJjTFLFqjgtfjkLIO7ty6H3lbsXIYSuUyxRaNbdXxkJcRUnyRoxW8cyPf"
            b"y4D1HAQvg+6mtlxW9Qcj7FxXLj0PpchBHPB4yvnNleC4lha0eqPH+02JyH/0r5D271aVpKodSVU7jnWpHleVowHStKocjoGgmob/"
            b"TcBf3dtUHghG3gcaOA4KM5+UrUFZL5hzoLlPHP4QsMVOxSKUJPtVbbyiillM3tDbrKGLL5W6Opkl3ZTun2k43e1LzvqNtD0WyzED"
            b"AwB2RUz7CX3szXGFkspQdZeF7hwHchwgzBqJLjgw7vN3EVDBKS1/FDQIpLhQ89UzrM5MCb6T9YpmtKz0o2WF1ZSVEKlsbE8oWUvE"
            b"dott5WRTTVadyNMm48a//4kq0h/Uil7ylybMxDuqRjiBK4nGsYwMTkIISRJaC+1jzmIUxXMR8RcCORuwNkoqp0TQ1wybJuUg+x62"
            b"pOOCb0SPv+hfOgFKzic8V5RCu3Q1otGfrGSP1Pi6hH2gLHjuUfr/7V1XguO4rt3Q+2AA03IYd3H3/hgUSIpUsF3V1T31MyO3IVlS"
            b"2RAInFA45FDPaMhmUzhquo7HNHUjUoyreF4PlEalfPs8WIPxECab71s+A0aAy2r2U3cluj9MOnYJZ3EFxfe/omw5nXAYMpX4jWr/"
            b"kuNp+qLdJ3XOJuMrThNXGPoWjDQSiMYnWbmmaLZbq6rfWpGebd6z5PvuOnW2/F/JmwdMfTUYWupXWtevUNevcLt+/Rb6vQYnEUle"
            b"AJ5yAAVeadAMa0HicyAE7wFJBYFZxjRWUsqYH31QNj4Sgkevm/exAb+z6ZOuBdYj2hM+fTI85TGN2epdWAqAgycVOVa3k4x2Ss9c"
            b"rUyvEjUc7KTS+wTQINc+9+GGtbjdkjfdeg2dAmsOoYu4CtmMB4v1jOwsunLcOzRP+XwANBvzdJ6izSIeqkX8HTOrC6++Zd3+4xPe"
            b"O5TOzxasjHrn4/8E5dJ5QgLSjiuQhgiqqMUavEMOK538S4wJgVgV4jYVRhj0a9d3f339SBzjHVuSo2lfcdn4rLlfuYycMinl1wB0"
            b"vvZncYnepbNH0CNaep+U7D0TNpP4yJ+fI2MRitBZWyI7Euagly37UMrZT5BDY8uRM+TQzEOq9ABIVWu2yKGdwQMnBMwKN/QBfOav"
            b"AEiXUDWSXimHnYl1gqV5pR/XdEa7AAo5zQ0HpbjB3lkKQjmhLDOSxFfKkO+DvI+m6c/yLL9Ktb0B1JpG1/5dP2O45cD8hqA/nq+i"
            b"+5FQfd/GQ6uqPDid8OBaMaCNZNv0fAFLsiVlbcB42onwbw4Ash1LsZEGaT5uCcaMCjYFWB72JKK0FIDuHQVWGbuIYS+Ilotu0E01"
            b"XLTv4ORzytEvSzN9qkh9dZ50F6W0E4jmW38nqegCJf89dKPBJEkHJYOXOpapEJfXBAeSTFGd1IooiwhJGHkmELLSEXBUxwW65khT"
            b"ExB48TtJ+p0k/axJ0mPdpdkkSU4nSfJ0kiRPepYHP7vj5u8k6dHC3BAds5dXCXfpwCmMveKOSMwIT6wfH3MYE8bHOtJrzJnQkhEr"
            b"hZGUMOx/J0l35xw32pV8Hw8RzFv1ITxUHzrkWjxGGZU8eehvHtPucWW+Z1WyKzkfhjizpsNQn6hcXvo7gaIAYwzRQRY/h+bkyzCd"
            b"Io9IY6Gd4nLSjlmtuaO8ZmF24iJkiY7/LWB/Mdrr2B1AsOzCQbBDl/VIUUg3uYS+NWDKX+an3qRjN9HzxDuWcDqzNamIm5UjyXjz"
            b"SOfc+J6VJfXviOlPjJgMjb8hSUPSEQXufPwfU0CoJs4aLeJjwTFEDKUuiBCCYkwJTrVnyvBYBv8qPD9TeL7XLr4nBP1Y4blRKNkb"
            b"0cMzTVGp2Kr8+qDybaavykdnZ77leQJM7ir/i3j0jEqaDpwvlCqkmocdbaWb+z5wCi8fFR/8K6V0Uy5pfHK3+oCLgnx/sb9LgWAh"
            b"yJNK+8oJaoHlL7/2+lWOJUVmaWUK1C8XIsGW7ttXTZ6ucvruPbX2gp85TP12fZ92fQ31lhLnOUfWQvwOciGJ1p4EwWJKdtxjBeAT"
            b"f5RIQV0s7ePi0uvgTErMX2OucqWA+o7L9FXvV31zX5cOXKV6WrwaWS+9ZdSK176mOAdkVZ2Nga3UI1/XyTDy0Eo/EQ/oY1en1vhf"
            b"OmoQDTm3ObgQ8RlHbNTp6RAMOSyrA7yYml+3aF1ycfuqqOLXubZ+hU9pWOcGrn/apvU7svMX9IK/zKPVQCyDKQcOhlGDFEHYUqw8"
            b"8zpWzBTHOllJ70AQYhCWgjGnkAuSYEWFfEm3uvw8v8Bde9YlfsNXu7fYq80+1EGMqrb1W8w6xll/d/WrKr2h1eCdSr439WO7W9St"
            b"R874WXLkr57kvKGnX+7tLHT/K0u/0pGEjqOfuatC7X/fpiWT3snd5Zeaxv81z78fYiW16lkcCf7XPeQPUlkNNwpYYFJjxrQQGgln"
            b"MYppD4g2AIaABIe1pIHGGpVrCR7HZ7q0TCDMv5HKCq+CEV6VKjlZ8X+qeB2aiG7JVe6mqQMjpYnyXg9MuyPTOaovZ66lSyFYVPSI"
            b"rGRX1gx3ENIjpWiqRfcmQqhZJjDHxW8jZzviQVXIguN+8U+YS1kJR9zFwIQwxcndWKW4Q50dPp9L2qNhtj5heD1S6xsrpcgpnuDs"
            b"nZmMaa3RtzdqZ1tQbf0F6inTwrOREGgKz2tVlS8CIZgkw8el0Cwu/rGwxlvrPXCLuUROMsG5cRQkgiQn5Xxq5sbfoJDIWi7cd7lB"
            b"QV9tPku953n3vhsUWwjuS1dxyVLDNt+tzMcLOFbSrQCc9lZxjpKMbh8O825mL4MAR6h+7VoFK29/4N56cK2CeY82nV1uwCKynaZc"
            b"0xt9/MjKh1k9o15QiHqooDdzgToTcC5RUOmZ7hQBqCgCMKYIdFp57b/+pZnuG/PZmsOsSUpRFAhxcUUMXDAsJeNx3eywQcih+I/K"
            b"Gso1cEJFXGdL7y2jNLj4WP2mGdNHmponHc0mi6XvTPXlqa/i9oKWLg26KpeLrV5pMfk4k6BY3UpUQ5T/pOMp7jRPzylOO4zzcZ6Z"
            b"U5wO2lcrwwkdLxW6Eb1gJfBF7fl4q564LY+Bo3jKbsJTdhOesptu8Dov2U0/tzX4jQ3ANW8FGZJxhjHcemw0eIo94tR7YTgJCqzF"
            b"JKUt7E0MA2U4JASVA8bAvEfFhzfg9a+uaCv1udO15dz37VpPfkTlvEtUuincfILv6udAUFaqApVrXmotdhx091D/W+TTK0mAgQhp"
            b"KwlwtlAtB824MkJGYnhHhftifpxX8hQxsfcF+DZaGqgls8WVHnMMbFXn3/doZAv6lXG8seX0QFG1nCTfVQAu9s4fWPZsYFRfhu5/"
            b"Bak/swG5MAi5uQWDrZ+pVVpl8gKaemUEf9SVKqtweq9q/RbElAWfQLUOcedjHqQ4LcZVEIwSp5xFVBOOKSApUoWrlIIQAALSHGED"
            b"7FsH9V/LxrouUd9K2RNHkVswo8Mjip/0Z/mqJ72Wqw0xC7f9gGLFhEsh2lOz8EX2P53KVzOzuZFLOVr+/HjvW4wY3rUzj1KAOTjd"
            b"EIYBQW18T6uHRmcRdf7EQGuPY/DEyIr8ahMqV1X0AGGbQ/P9x7DZBa77DM6rRJUr2R4Njwv4+KV7pqs1dTqdeuvJBjR7750Ch22z"
            b"fZ59dVtz5a1/mZtbTcdGj4jvx2+NHhEiJimjiQAZgkESAPHEHtPBxEeFB4MUFSQ+D3wgVgdtKReYOWs9jb849Xf7q3wQVHvz8TGZ"
            b"xPW+L18Dqr2Le32k3XCK1D34VF8hdbsd3oTSptPHnGC6GlXLStN2DA3O0bnx/J7jChdCkEeNlzNcLT7F1eILXC0+xdU2oK0NV4s3"
            b"XC3ecLW1QtePwdX+Wy4s1nHMmIl1OoJgiKfBMasdDQJQkEpbiN9KpikKlDtiaGDYg1DIMGCWqf8ceCuLFLAK96l6blp6lzBVuYr0"
            b"uTbrF2B+aHVdSjUOq365dUZyUlY1xe0QzIqTYPwyMrUbxOzI2f5xpUoglYy13epBg6btjjcw5iVJpqOkz1alKBVrl0NcQZCPELR8"
            b"jCyi8CJ0Kz0+n7B9Z859cpmWrciAenvsaiWnTWq51bdyq2VrbsEViuAXn1Xjs2wAIyF+57gUIYBFxmmrTQhSOCEoVhw0i68IkpIo"
            b"SZgJyX1F6pjkkGT/sFbMbTLAh0RlruVfbiEOvlrQ5ZGiTRF+SbalK1eLLQcXrVbtEdTF5OqOSni7I5YbNfeQPc+NCPvgfOw0aqQU"
            b"8cPdOlr/FTWbHPwqgSB9oZ71lCF/wXYHVVl5+8lKr7XW4q4oBs0L9n9LDbonad52FEYuBLXfwA9X7/ondGMciqnXc8+IYcw6EwtP"
            b"hSwoFWKtQRhKnC0UC1KlwGNMvAPsOCeMYqkNMn8YM9vMEPmlv1VfTt2c6PcY1k4R4qEjyrQBcCz2Rin4pmn1MZuMTKuH6/l60Lg1"
            b"TkU99zqqWZyBbo+2qaqsTSur662nexSXyKec7kGsVsV+7mcK5IW3mx2yCSPLZ8jVTYtU+Vy11bBYdmhdXx66a9HHSXdsmzqzumZT"
            b"Q+uzdyqLl87tpR/bHc1gfo6fVtN9nRi/nAzoOuOXS6etr0u7mEvOvQlaOsaE0IoYBBB/RNo7rgMO1IDxqeUaF/UY4mvA2FrHA40V"
            b"8beu8q94ta+Zrr4uq9jStY4Dmq/t6c4mfZ1ZYnNXno/5Jjq3R00dvqdeysj+W1zGa0dcRe5qlfaE6rwdJxSD7WGwKh/wqqQeGDSk"
            b"A6d4ujRdltOvXRHYoMXA1HINmHGMD9fdIeJKkBDk++i1cu2Z1iyu7cWJLg2vscD1i0WwoOaMNS9kozmz/huHfrMWqln5Y3/vIO0t"
            b"Su3r7Qqat+m4deEsFkwHBNYRrphxyknjhdc+iRswA0km3Adkgw0EI821oFZaQRLGLqgf7o1NrnxjXyit94Q4SYdbIt+K6r6ibmZg"
            b"dJ2uH1fwtyvqBVRAKkgB3cEEy5H7KhyV/rDAfLjTwJpww2iIdSynuuhWHyzHlaeRkLLdo7UwTG+nxxsg2C9hqBJOoQStfNxFv+lL"
            b"7GTlp4Brs0qYVHHkSb37Z82zP1HXvucU+1H/bGdjgSpASpF8sb1wTBpkMDKIUMOYVJRT4a3mnruYI4VBXMWzMrF2FUrIr5Bo/IjN"
            b"y4Vy4yVe4BIhPBJ2vN0uOAJt6eLy2v/iU0BKqbglFCw/5rOKsVZ37Af2pSQjWQKr9DoPaoeNGAxBS+TRnGCoPnjXnuuuxORid3AQ"
            b"aRnaii8iLVzWlrCnQjMf8YV9pOlCqukW6VxdSTW1alllM9LFbAaGt+RYb+3OWvvcvyJofIJb9qfsZ953hf0y/RaPgfOYZ5V2RmHH"
            b"uEKea0804pJKCRR7C9jiYLVyRCJEUJCZYpucZOGV5kBuDcwKzle4H292Bz6mc/Ae2+O5zgFvuCgwx16NWRh3GxNnjozHFXx5NODV"
            b"Sms9x6FuwaiVi68kFLpu7jmzY+nMFtWFXUZh51gMxBTSua9iChzDegv5Rs7occA56l+SOBjJGdyRPfiLSRdvSBx8WSPA4+Ru40Fa"
            b"zhSWFjy1yBtvDfMs/haspggHZmgAJQXSDAcuQ8zVJAlvyC/AMEyAWWwOsGIXACs5XscXtGgNVBl2Zed0gHu2BXQzzipz8lTsqWoQ"
            b"D10zM4XF/EC3PCKqWoKet4Y3ke8WEFDgALIAsnaGHT0CcCFHKY4q165d0Lt5MKUgVT53ozMPQyGHKUG6K+rHdOmaVaVn8BBw9agI"
            b"nckZ4KmcwRqFKzmDzii2lv4+coLHDrAfkAv8KOQq73YFAnhkq/BRNQNPAfFYJgJCBiwh2mChrdNSBuywpIkT7IQPjgWMiQakbUxb"
            b"WFhnpMXis6aszwwVzt0U3mbtDj2abroy3DFgvXDB7nBRCbZUE3vZWvcN9xhNh442sQcHQlI5DRxJwgPHxOMxM44w2Q+O7bq7o2YX"
            b"h5WRxfGKeuJLMwC2gq+b3eej79avfDd+YEOP7HzwHP/WsvyFPmarcnAl8DJbfF96wN4l235uav/zPAr+nBPBlks1JQLxuCzHzsm4"
            b"/MYGE+RCzKLGcOx1LAuBBaqtRDGOYh4QtkgHrSHQv9t94J5lwCdH77cYUDdJVy+4D9wiuN7kU+WPzWmTCD5is06cXh/RsLJQbeY7"
            b"EUplncdnM6JCmE3BOPl2tkTWySnlwHwZr5sYKpHy+8OxezNPbzD9vM6cvC45+fr2MnknnQar3PN0/eLMZ+BoGFOxAfLs/ddp4KNJ"
            b"l1sC3AvPkTQMWSMkFbGuUMhQ4p1CgYPA1ATDmNIaYxCW0qCUxtwF8k/Icc0G63hdkh8FAyfZU7Rr57TIZeuRNoXpo4rWMBte6eyI"
            b"w6n3WKKa/ElfQebmWrE4HAIZu7O00FtWImOpKOrnE51AVe/omqVDZSmIWpnrKVk0fciTluN4UL4syyucPmmcDskGT9q35eLRjbd3"
            b"oG058pPNnyLQdcEAPctqX8nzPCYzwRx3Oi7BZQjgECjtkSRUSA2WE5okujCWTGFnvRJxKY4MNyAQDUgy+69zO3vkzZDbiS+5nfgh"
            b"CzLtn0iQdEHGi9Kf27BG7QG/ggQKCwUUyUbrBE/Cc1xKGVPOKL4ib3a3Oh8oe7iIdYiz3oWMi+ohTule5dCm9/g066U/71PG5xHq"
            b"I6cT8ncZn/t45ccxPq8473MI5QJEgmHOyyf3WdKnV85RB4lZJCggy5VFRgIn8aerjQkqLqQdEfFtimWM8sHhIL33goZY3v1qQk81"
            b"oYuoA6slnGsCovjzEs63SKSd1Yms1p/DmUu84hT6Q2ShH7qezuYocto1XOWWoEpPLbn9KAs986SuSrdfWeiHgxSNQ3IXTRlKQwDr"
            b"peNaOO04UZJqqrgO3ODkN0oM0Vqq4LxnHBiWAb5CXvVNhPbN1tk7mqgHSYrbZtbPWl9nhtVDTuGV8/OGBz86P+dLHNs4j4HYVzbO"
            b"cjXrYKcTpiEccTIIOvppT+c7r1hvD/yjy8UxSije95IV5ry7vhJZ/rit6/RD7PlDKCVU5SAc5jcjwOSs7MTTCdAYStnpKlWTbRhs"
            b"fRBg+VGc+myS02gsTSY5m8bSpdv0Ja79W4TxfKpOqVAQ03rwPC7ZYyonwhtlDXdxAR9k8JI5Ki0z8StMeeBWe8aEdkyq75MroYN1"
            b"eg/4uRJVupipj4Eyt/A896BBExp7d6y2yVhEROmYhBgPlJBEy1yDbFSZAYiod1yS1dTzAESMx0uRVKpd+fPEb+k2g7Tt2G4CKkcl"
            b"ES4r/D7f2VF8w3bdg1uqkYx1vqwFaLnOdx7iih4bpYxblmcWKjPxp3MLFXKYox+Z7q34SNn6O7FGN9oAX49CGjDfvXEYSRsoIxgF"
            b"6mOZLK0KHFGDguVgNGijPDdUCkQcGIcAYcelIwTcl7CInuGSxBU06TFj6Cbq6KDnvNOX9r9shWEshTYlK46Rr9mhhDeApnTIEkkZ"
            b"QnWayl/MnMGbfmEKKvU8w1SRIS+oO6WDcnUt4dGG3rMsuEsdSh+YuqDLQypp5i0fDcezTEErZonyI5OqDU/nCA/q6nzIJAjdco3q"
            b"3vDgzuX4cp9jsUzYNU+phOHOi+ApIOpJg3ZDdfIa1snrWpnXxfL+AqrRe719SVY6gENXxjzecny9tTZ2azLTVYb/K5FSl/X1H8BQ"
            b"jeprH5CnCnRAnHnpUnVNnIzfXOkREQaLYIm0xirDsCNWI2swUyLGcsDyr3Os+bwlzAf9avBO+1l1QaY2rDedd243J1JIqr/RHqim"
            b"vQa0tXoUJf0Ox7J6BILd+jRHKQFa6uT02Nz+2hnZu7V2Bm0hlR2cYzxLGpbNOY1BVPke5wYUXVmqZ4JcJawcPOZzhHcx69l6qoRR"
            b"eNWL5hk3YMxmwlM209k786PtmbwlqM5Jq11Ppdr6taL5diuaQKkgjnOMmRTSgPdKeIxczPgScQyeIUkSxhYpFBgwbKzhQnqMDWMY"
            b"/nPC1q3MFX2scSWX9MvXgA1yytrpX3KRFunJhFagRRV3xHnlPvA2rqvC+UgW+i5eArYG0A37mRyVMnsP5y09kO4U0pnmyFULBW+o"
            b"ubYFtQihCLX/lVvJBpGz9ot8q9UQ4FFzhOeFy45gpXWzI1n47nqr7StWQrcSvH1VGiUFr7phvfj+btnGuRgsqz+yvSq6/4sm60fk"
            b"pv5LGtghlrCEeLDBaaIki7Wt8sJapQGclAGQtIAsM9wzyrlQAjkcrLXBE8Plj8qCYp4I8VUufN6SHmSSYVf6gWZfwZc1TlFd70Zk"
            b"FNr73gEsA8wZgRquRZYUe5D4TycrWIXDWimtbaQoP80Kud9TX3sl1VX7jwo1bV53kllzfcF5u7kXjKmuddx2TjdmUY2hqFLn2oO7"
            b"K0kXUEJfbVM/IPmfEVbn48H5O+eKLn2z4mpw+MOa1LP8W5rXz+T+vjIzn7L8g9GIIBZUoDIEqQkwz4TFPomscIcAvKGOYWWCFYhb"
            b"FWwsbg3yzLtYwv7HatXGIRyq61itv8vvflsXizXt1MdQGSmBm9XznpqaujanXLLiaqtatU/NMeZOys05Ln3xScXiH/KDc55SrRgg"
            b"qdLa0bFF5LY7X+n8bbNi/CF3S2ZV6mWyqQHW9402D59VC/ClwlV+qBcw1vB7tUtwve7/LUnvlqRWIkqlw9QTpaizHmlOCKbgcfx4"
            b"Hr/1RFCrnXVAfCxQmKXWeZmosIz/ixO5QYtzKPqPl0ozWdbs6XbRflraLw8lCJ6A5EqtG/8j2o4tpa+0YU/1AgYQi0mrVNW43a5s"
            b"IxtSb4yg2/dtE/4KoCvJfGsOswrx3O0kCkqPjJqyMN8t38v81zz2ZsdusDmscG9pJai9tluGn1Jit4buGwM6BPIzrrBQOWXV23un"
            b"oGkUbKAK3sz86lcjd9cOabFYGPawi5VH25nFLk0J/juu+/5x3f/+H1tPWAg="
        )
    ).decode("utf-8")
)
_LABELED_2X2_BUILTIN_CACHE_DIGESTS = frozenset(_LABELED_2X2_BUILTIN_CACHE_ENTRIES)
_LABELED_2X2_PERSISTENT_CACHE_PATH: str | None = None
_LABELED_2X2_PERSISTENT_CACHE_ENTRIES: dict[str, dict[str, Any]] = {}
_LABELED_2X2_PERSISTENT_CACHE_DIRTY = 0
# The exact subset DP scales as O(3**n), so proactive tree optimization is
# disabled by default.  Cache misses use the cheap exact greedy tree whenever
# it satisfies the hard memory cap; DP remains mandatory when greedy cannot.
# Repeated-run tuning can explicitly opt in for networks up to N operands via
# TANGENT_BP_BOUNDED_2X2_OPTIMIZE_MAX_OPERANDS=N (0..16).
_LABELED_2X2_DEFAULT_OPTIMIZE_MAX_OPERANDS = 0
_EINSUM_PATH_REFERENCE_BATCH = 2
_GATE_FACTOR_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_STAGE_PROFILE_EVENT_SINK: Any | None = None


def _set_stage_profile_event_sink(sink: Any | None) -> None:
    """Install an opt-in, non-synchronizing benchmark event sink.

    Production runs leave the sink unset, so the only hot-path cost is one
    ``None`` check at each coarse profiling boundary.  The sink is private and
    deliberately carries metadata only; it must not inspect or modify tensors.
    """
    global _STAGE_PROFILE_EVENT_SINK
    _STAGE_PROFILE_EVENT_SINK = sink


def _bounded_2x2_optimize_max_operands() -> int:
    """Return the explicit proactive-DP limit; zero is the fast default."""
    raw = os.environ.get("TANGENT_BP_BOUNDED_2X2_OPTIMIZE_MAX_OPERANDS")
    if raw is None:
        return _LABELED_2X2_DEFAULT_OPTIMIZE_MAX_OPERANDS
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            "TANGENT_BP_BOUNDED_2X2_OPTIMIZE_MAX_OPERANDS must be an integer in 0..16"
        ) from exc
    if value < 0 or value > 16:
        raise ValueError(
            "TANGENT_BP_BOUNDED_2X2_OPTIMIZE_MAX_OPERANDS must be in 0..16"
        )
    return value


def _emit_stage_profile_event(
    name: str,
    phase: str,
    **metadata: Any,
) -> None:
    sink = _STAGE_PROFILE_EVENT_SINK
    if sink is not None:
        sink(name, phase, metadata)


@dataclass(frozen=True)
class SingleSiteConfig:
    """Numerical and memory controls for the single-site engine."""

    # PEPS truncation and exact scheduling controls.
    chi_max: int = 64
    # ``randomized`` computes only chi_max plus a small oversampled spectral
    # subspace for compression SVDs.  The exact Frobenius norm of the input is
    # still used to certify the discarded tail at the selected rank.  The
    # historical full SVD remains the default for compatibility.
    svd_solver: str = "full"
    svd_oversample: int = 16
    svd_power_iterations: int = 2
    svd_random_seed: int = 0
    branch_batch_size: int = 16
    two_qubit_apply_batch_size: int = 8
    two_qubit_layer_gate_batch_size: int = 32
    bp_branch_chunk_size: int = 8
    readout_branch_chunk_size: int = 8
    # Concurrent source-branch chunks for the *same* target site.  Each CUDA
    # stream contracts an independent branch range against an identical
    # support=(site,) network, so the Cotengra expression/path cache is shared
    # while the accumulated one-site RDM remains only 2x2 per branch.
    readout_branch_stream_count: int = 1
    # Concurrent independent support=(site,) RDM contractions. This adds a
    # scheduling dimension only; it never constructs a multi-site support.
    readout_site_chunk_size: int = 1
    # ``tile_major`` releases each propagated source-shape tile after reading
    # every target. ``target_major`` retains the completed source-layer bank,
    # then exhausts every source tile for one fixed target before moving to
    # the next target. The latter enables same-target branch-stream reuse.
    pauli_readout_schedule: str = "tile_major"
    readout_patch_batch_size: int = 16
    bethe_site_batch_size: int = 32
    batch_disjoint_two_qubit_gates: bool = True
    reuse_opposite_bp_cavities: bool = True
    batch_readout_patches: bool = True
    batch_bethe_sites: bool = True
    # If a cached/preflight contraction path cannot satisfy the configured
    # memory limit, split first over compatible patches and then over source
    # branches.  Every successful tile still contracts the identical tensor
    # network; this changes scheduling only.
    auto_readout_tiling: bool = True
    # Project the identity and target operators inside the readout cluster so
    # the physical density matrix is not materialized.  The fast path is used
    # only when raw-matrix conditioning ledgers are disabled; otherwise the
    # historical raw transition remains authoritative.
    fuse_readout_operators: bool = False
    # For an immediately following pure 1Q ideal layer, project its target
    # operators through the preceding 2Q readout.  This reuses the identical
    # frozen virtual environment and skips the covered 1Q contractions.  The
    # fast path is deliberately narrow and falls back for unmatched sites.
    reuse_one_qubit_readout: bool = False

    # Source/target Kraus representation. Nonzero filters are approximations.
    # Keep all source modes by default.  A nonzero value is an explicit
    # approximation and must be selected by the caller.
    source_amplitude_filter_tol: float = 0.0
    target_amplitude_filter_tol: float = 0.0
    source_kraus_gauge: str = "overlap_balanced"
    source_kraus_gauge_phase: str = "zero"
    source_kraus_gauge_seed: int = 0

    # Single-site fixed-point solve.
    bp_max_iter: int = 200
    bp_tol: float = 1e-10
    bp_damping: float = 0.5
    bp_residual_check_interval: int = 4
    # A strict fixed-map pass is still required before convergence is
    # accepted.  Scheduled passes are deferred while the inexpensive sweep
    # step is clearly far from the fixed point; the final pass is never
    # deferred.  ``None`` restores the historical unconditional schedule.
    bp_step_residual_gate_factor: float | None = 100.0
    bp_active_compaction_ratio: float = 0.5

    # Local readout estimator and failure diagnostics.
    branch_weight_mode: str = "bethe_ratio"
    # ``cluster`` retains the historical support/strip/2x2 local readout.
    # ``gloop`` keeps the streamed PEPS, source propagation, and BP solve
    # unchanged, but replaces only the final local contraction by Quimb /
    # Cotengra generalized-loop cluster expansion.  Quimb never sees or
    # contracts the full lattice in this mode.
    local_readout_method: str = "cluster"
    gloop_size: int = 8
    gloop_combine: str = "sum"
    gloop_grow_from: str = "alldangle"
    gloop_optimize: str = "auto-hq"
    gloop_expression_cache_size: int = 4096
    # Optional exact, memory-aware Cotengra planning. The ordinary optimizer
    # is tried first. Only trees whose largest intermediate exceeds this
    # per-contraction budget are re-searched with ``minimize="size"`` and,
    # if still necessary, exactly sliced over contracted indices.
    gloop_memory_target_gib: float | None = None
    gloop_memory_search_repeats: int = 32
    gloop_memory_max_slices: int = 64
    # Opt-in conservative concurrent-memory planning + exact serial slicing.
    # The budget is not a process-wide CUDA hard cap (allocator cap is separate).
    gloop_peak_budget_gib: float | None = None
    gloop_peak_reserve_gib: float = 8.0
    gloop_peak_safety_factor: float = 2.0
    gloop_oom_max_retries: int = 6
    # The small trace floor is only a safety diagnostic for conditional
    # readouts.  The unnormalized full-Bethe path never divides by trace.
    trace_floor: float = 1e-30
    auto_skip_nonconverged_pairs: bool = False
    conditional_kappa_warn: float = 1e2
    conditional_kappa_skip: float = 1e3
    auto_quarantine_ill_conditioned: bool = False
    # ``branch`` keeps stable source-mode contributions in the numerator and
    # routes only unresolved source-mode/readout-support combinations to the
    # fallback queue.  ``pair`` is retained as a compatibility mode for the
    # historical whole-pair exclusion behavior.
    quarantine_granularity: str = "branch"
    retain_quarantine_ledger: bool = True
    # ``strip`` uses a two-site readout anchor for every target.  The actual
    # one- or two-site target support is unchanged: for a 1Q target the other
    # anchor site's physical leg is traced.  The two long sides receive
    # correlated adjacent 2x1 boundary strips, while the two one-leg ends use
    # ordinary 1x1 BP cavity messages.  Readout policy is selected per source
    # computation, not per target arity: production source0 uses ``support``;
    # production source1 currently uses ``2x2``.
    # DEFAULT FOR 2Q-SOURCE PRODUCTION: every later target, irrespective of
    # target arity, is embedded in a central 2x2 region.  Together with the
    # default ``readout_boundary_mode="joint_halo"``, its four available sides
    # are closed by four correlated 2x1 joint strips.  For a 1Q target only
    # the actual target physical leg remains open; the other three are traced.
    readout_block: str = "2x2"
    # ``single_site`` gives a standard Bethe region: the raw cluster contains
    # exactly the readout patch and every crossing edge receives one ordinary
    # incoming BP message.  ``joint_halo`` retains the historical correlated
    # boundary ablation.
    readout_boundary_mode: str = "joint_halo"
    # ``matrix_free`` is an exact 2x2 path that keeps joint-halo sites as
    # factors and closes all virtual indices before the small target-operator
    # contraction. ``einsum`` materializes the same exact local transition.
    readout_contraction_mode: str = "einsum"
    # Explicit opt-in approximation of each correlated joint-halo boundary.
    # ``None`` preserves the exact matrix-free network.  A positive rank uses
    # independent, reproducible randomized factorizations for every branch and
    # boundary direction without materializing the dense two-leg halo tensor.
    joint_halo_rank: int | None = None
    joint_halo_oversample: int = 16
    joint_halo_power_iterations: int = 1
    joint_halo_random_seed: int = 0
    # When set, every converged readout branch must satisfy
    # outside_factor * identity_cluster == global_Bethe_Z to this relative
    # tolerance before any target insertion is evaluated.
    strict_region_closure_tol: float | None = None
    record_support_consistency: bool = False
    einsum_memory_gib: float = 2.0
    retain_pair_ledger: bool = True

    def validate(self) -> None:
        integers = {
            "chi_max": self.chi_max,
            "svd_oversample": self.svd_oversample,
            "branch_batch_size": self.branch_batch_size,
            "two_qubit_apply_batch_size": self.two_qubit_apply_batch_size,
            "two_qubit_layer_gate_batch_size": (self.two_qubit_layer_gate_batch_size),
            "bp_branch_chunk_size": self.bp_branch_chunk_size,
            "readout_branch_chunk_size": self.readout_branch_chunk_size,
            "readout_branch_stream_count": self.readout_branch_stream_count,
            "readout_site_chunk_size": self.readout_site_chunk_size,
            "readout_patch_batch_size": self.readout_patch_batch_size,
            "bethe_site_batch_size": self.bethe_site_batch_size,
            "bp_max_iter": self.bp_max_iter,
            "bp_residual_check_interval": self.bp_residual_check_interval,
            "joint_halo_oversample": self.joint_halo_oversample,
            "gloop_size": self.gloop_size,
            "gloop_expression_cache_size": self.gloop_expression_cache_size,
            "gloop_memory_search_repeats": (self.gloop_memory_search_repeats),
            "gloop_memory_max_slices": self.gloop_memory_max_slices,
        }
        for name, value in integers.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.svd_solver not in {"full", "randomized"}:
            raise ValueError("svd_solver must be full or randomized")
        if (
            isinstance(self.svd_power_iterations, bool)
            or not isinstance(self.svd_power_iterations, int)
            or self.svd_power_iterations < 0
        ):
            raise ValueError("svd_power_iterations must be a nonnegative integer")
        if (
            isinstance(self.svd_random_seed, bool)
            or not isinstance(self.svd_random_seed, int)
            or self.svd_random_seed < 0
        ):
            raise ValueError("svd_random_seed must be a nonnegative integer")
        for name, value in {
            "batch_disjoint_two_qubit_gates": (self.batch_disjoint_two_qubit_gates),
            "reuse_opposite_bp_cavities": self.reuse_opposite_bp_cavities,
            "batch_readout_patches": self.batch_readout_patches,
            "batch_bethe_sites": self.batch_bethe_sites,
            "auto_readout_tiling": self.auto_readout_tiling,
            "fuse_readout_operators": self.fuse_readout_operators,
            "reuse_one_qubit_readout": self.reuse_one_qubit_readout,
        }.items():
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean")
        if self.bp_step_residual_gate_factor is not None and (
            not math.isfinite(self.bp_step_residual_gate_factor)
            or self.bp_step_residual_gate_factor <= 0.0
        ):
            raise ValueError(
                "bp_step_residual_gate_factor must be positive and finite or None"
            )
        if not isinstance(self.readout_block, str) or self.readout_block not in {
            "support",
            "strip",
            "2x2",
        }:
            raise ValueError("readout_block must be support, strip, or 2x2")
        if self.local_readout_method not in {"cluster", "gloop"}:
            raise ValueError("local_readout_method must be cluster or gloop")
        if self.pauli_readout_schedule not in {"tile_major", "target_major"}:
            raise ValueError(
                "pauli_readout_schedule must be tile_major or target_major"
            )
        if self.gloop_combine not in {"sum", "prod"}:
            raise ValueError("gloop_combine must be sum or prod")
        if self.gloop_grow_from not in {"all", "any", "alldangle", "anydangle"}:
            raise ValueError(
                "gloop_grow_from must be all, any, alldangle, or anydangle"
            )
        if not isinstance(self.gloop_optimize, str) or not self.gloop_optimize:
            raise ValueError("gloop_optimize must be a nonempty string")
        if self.gloop_memory_target_gib is not None and (
            not math.isfinite(self.gloop_memory_target_gib)
            or self.gloop_memory_target_gib <= 0.0
        ):
            raise ValueError(
                "gloop_memory_target_gib must be positive and finite or None"
            )
        if (
            self.local_readout_method == "gloop"
            and self.branch_weight_mode == "bethe_unnormalized"
        ):
            raise ValueError(
                "gloop readout requires a conditional branch weight mode; "
                "bethe_unnormalized depends on a single Bethe region factor"
            )
        if self.gloop_peak_budget_gib is not None:
            from residual_tn.backend.bounded_gloop import MemoryPolicy, GIB

            if self.gloop_memory_target_gib is None:
                raise ValueError("peak-bounded gloop requires an intermediate budget")
            MemoryPolicy(
                total_bytes=int(self.gloop_peak_budget_gib * GIB),
                intermediate_bytes=int(self.gloop_memory_target_gib * GIB),
                reserve_bytes=int(self.gloop_peak_reserve_gib * GIB),
                safety_factor=self.gloop_peak_safety_factor,
                max_slices=self.gloop_memory_max_slices,
                max_retries=self.gloop_oom_max_retries,
                search_repeats=self.gloop_memory_search_repeats,
            ).validate()
            if (
                self.readout_site_chunk_size != 1
                or self.readout_branch_stream_count != 1
            ):
                raise ValueError(
                    "peak-bounded gloop requires serial site/branch streams"
                )
        if (
            self.local_readout_method == "gloop"
            and self.strict_region_closure_tol is not None
        ):
            raise ValueError(
                "strict_region_closure_tol is defined only for one local "
                "Bethe cluster, not a gloop cluster expansion"
            )
        if not isinstance(
            self.branch_weight_mode, str
        ) or self.branch_weight_mode not in {
            "bethe_ratio",
            "source_anchored",
            "source_anchored_sumratio",
            "bethe_unnormalized",
        }:
            raise ValueError(
                "branch_weight_mode must be bethe_ratio, source_anchored, "
                "source_anchored_sumratio, or bethe_unnormalized"
            )
        if self.readout_boundary_mode not in {"single_site", "joint_halo"}:
            raise ValueError("readout_boundary_mode must be single_site or joint_halo")
        if self.readout_contraction_mode not in {"einsum", "matrix_free"}:
            raise ValueError("readout_contraction_mode must be einsum or matrix_free")
        if self.joint_halo_rank is not None and (
            isinstance(self.joint_halo_rank, bool)
            or not isinstance(self.joint_halo_rank, int)
            or self.joint_halo_rank < 1
        ):
            raise ValueError("joint_halo_rank must be a positive integer or None")
        if (
            isinstance(self.joint_halo_power_iterations, bool)
            or not isinstance(self.joint_halo_power_iterations, int)
            or self.joint_halo_power_iterations < 0
        ):
            raise ValueError(
                "joint_halo_power_iterations must be a nonnegative integer"
            )
        if isinstance(self.joint_halo_random_seed, bool) or not isinstance(
            self.joint_halo_random_seed, int
        ):
            raise ValueError("joint_halo_random_seed must be an integer")
        if self.joint_halo_rank is not None and (
            self.readout_contraction_mode != "matrix_free"
            or self.readout_boundary_mode != "joint_halo"
            or self.readout_block != "2x2"
        ):
            raise ValueError(
                "joint_halo_rank requires matrix_free 2x2 joint_halo readout"
            )
        if self.strict_region_closure_tol is not None and (
            not math.isfinite(self.strict_region_closure_tol)
            or self.strict_region_closure_tol <= 0.0
        ):
            raise ValueError(
                "strict_region_closure_tol must be positive and finite or None"
            )
        if not math.isfinite(self.trace_floor) or self.trace_floor <= 0.0:
            raise ValueError("trace_floor must be positive and finite")
        if not isinstance(self.record_support_consistency, bool):
            raise ValueError("record_support_consistency must be boolean")
        if not isinstance(self.auto_skip_nonconverged_pairs, bool):
            raise ValueError("auto_skip_nonconverged_pairs must be boolean")
        if not isinstance(self.auto_quarantine_ill_conditioned, bool):
            raise ValueError("auto_quarantine_ill_conditioned must be boolean")
        if self.quarantine_granularity not in {"branch", "pair"}:
            raise ValueError("quarantine_granularity must be branch or pair")
        if not isinstance(self.retain_quarantine_ledger, bool):
            raise ValueError("retain_quarantine_ledger must be boolean")
        if (
            not math.isfinite(self.conditional_kappa_warn)
            or self.conditional_kappa_warn <= 0.0
        ):
            raise ValueError("conditional_kappa_warn must be positive and finite")
        if (
            not math.isfinite(self.conditional_kappa_skip)
            or self.conditional_kappa_skip <= 0.0
            or self.conditional_kappa_skip < self.conditional_kappa_warn
        ):
            raise ValueError(
                "conditional_kappa_skip must be finite, positive, and >= "
                "conditional_kappa_warn"
            )
        if not 0.0 <= self.bp_damping < 1.0:
            raise ValueError("bp_damping must lie in [0, 1)")
        if (
            not math.isfinite(self.bp_active_compaction_ratio)
            or not 0.0 < self.bp_active_compaction_ratio <= 1.0
        ):
            raise ValueError("bp_active_compaction_ratio must lie in (0, 1]")
        if not math.isfinite(self.bp_tol) or self.bp_tol <= 0:
            raise ValueError("bp_tol must be positive and finite")
        if (
            not math.isfinite(self.source_amplitude_filter_tol)
            or self.source_amplitude_filter_tol < 0
        ):
            raise ValueError(
                "source_amplitude_filter_tol must be finite and nonnegative"
            )
        if (
            not math.isfinite(self.target_amplitude_filter_tol)
            or self.target_amplitude_filter_tol < 0
        ):
            raise ValueError(
                "target_amplitude_filter_tol must be finite and nonnegative"
            )
        if self.source_kraus_gauge not in {
            "original",
            "overlap_balanced",
        }:
            raise ValueError(
                "source_kraus_gauge must be 'original' or 'overlap_balanced'"
            )
        if self.source_kraus_gauge_phase not in {
            "zero",
            "random",
            "fourier",
        }:
            raise ValueError(
                "source_kraus_gauge_phase must be 'zero', 'random', or 'fourier'"
            )
        if isinstance(self.source_kraus_gauge_seed, bool) or not isinstance(
            self.source_kraus_gauge_seed, int
        ):
            raise ValueError("source_kraus_gauge_seed must be an integer")
        if not math.isfinite(self.einsum_memory_gib) or self.einsum_memory_gib <= 0:
            raise ValueError("einsum_memory_gib must be positive and finite")


from residual_tn.backend.kraus import compute_dressed_kraus


def build_overlap_balancing_unitary(
    amplitudes: torch.Tensor,
    phases: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a unitary that makes ``U @ amplitudes`` equal-magnitude."""
    if amplitudes.ndim != 1 or amplitudes.numel() == 0:
        raise ValueError("amplitudes must be a nonempty rank-1 tensor")
    if not amplitudes.is_complex():
        raise ValueError("amplitudes must have a complex dtype")
    if bool((~torch.isfinite(amplitudes)).any()):
        raise ValueError("amplitudes must be finite")
    norm = torch.linalg.vector_norm(amplitudes)
    if (not bool(torch.isfinite(norm))) or float(norm.item()) == 0.0:
        raise ValueError("cannot balance an all-zero amplitude vector")

    count = int(amplitudes.numel())
    if phases is None:
        phases = torch.zeros(
            count, dtype=amplitudes.real.dtype, device=amplitudes.device
        )
    else:
        phases = torch.as_tensor(
            phases, dtype=amplitudes.real.dtype, device=amplitudes.device
        )
        if phases.shape != amplitudes.shape:
            raise ValueError("phases must have the same shape as amplitudes")
        if bool((~torch.isfinite(phases)).any()):
            raise ValueError("phases must be finite")

    source = amplitudes / norm
    target = torch.exp(1j * phases).to(amplitudes.dtype) / math.sqrt(count)

    def _complete(first: torch.Tensor) -> torch.Tensor:
        pivot = int(torch.argmax(first.abs()).item())
        identity = torch.eye(count, dtype=first.dtype, device=first.device)
        helper_ids = [index for index in range(count) if index != pivot]
        seed = torch.cat((first.unsqueeze(1), identity[:, helper_ids]), dim=1)
        basis, _ = torch.linalg.qr(seed, mode="reduced")
        basis[:, 0] = first
        return basis

    source_basis = _complete(source)
    target_basis = _complete(target)
    return target_basis @ source_basis.conj().transpose(-1, -2)


def rotate_kraus_modes(
    modes: torch.Tensor,
    unitary: torch.Tensor,
) -> torch.Tensor:
    """Apply a unitary change of basis to the leading Kraus-mode axis."""
    if modes.ndim < 1 or unitary.ndim != 2:
        raise ValueError("modes and unitary have invalid ranks")
    if unitary.shape != (modes.shape[0], modes.shape[0]):
        raise ValueError("unitary shape must match the Kraus-mode count")
    return torch.einsum("au,u...->a...", unitary, modes)


def _source_gauge_phases(
    count: int,
    *,
    kind: str,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if kind == "zero":
        return torch.zeros(count, dtype=dtype, device=device)
    if kind == "fourier":
        return (2.0 * math.pi / count) * torch.arange(count, dtype=dtype, device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    phases = 2.0 * math.pi * torch.rand(count, dtype=torch.float64, generator=generator)
    return phases.to(dtype=dtype, device=device)


def _source_gauge_data(
    data: dict[str, Any],
    gate_id: int,
    config: SingleSiteConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    amplitudes = data["a_modes"].reshape(-1)
    modes = data["E_modes"]
    if config.source_kraus_gauge == "original":
        return modes, amplitudes, None
    phases = _source_gauge_phases(
        int(amplitudes.numel()),
        kind=config.source_kraus_gauge_phase,
        seed=config.source_kraus_gauge_seed + gate_id,
        dtype=amplitudes.real.dtype,
        device=amplitudes.device,
    )
    unitary = build_overlap_balancing_unitary(amplitudes, phases)
    return rotate_kraus_modes(modes, unitary), unitary @ amplitudes, unitary


def _build_neighbors(width: int, length: int) -> dict[int, dict[str, int]]:
    neighbors: dict[int, dict[str, int]] = {}
    for site in range(width * length):
        row, col = divmod(site, width)
        local: dict[str, int] = {}
        if row:
            local["U"] = site - width
        if row + 1 < length:
            local["D"] = site + width
        if col:
            local["L"] = site - 1
        if col + 1 < width:
            local["R"] = site + 1
        neighbors[site] = local
    return neighbors


def _zero_peps(ctx: Any) -> dict[int, torch.Tensor]:
    peps = {}
    for site in range(int(ctx.width) * int(ctx.length)):
        value = torch.zeros((1, 1, 1, 1, 2), dtype=ctx.dtype, device=ctx.device)
        value[0, 0, 0, 0, 0] = 1
        peps[site] = value
    return peps


def _reference_batch_operands(
    equation: str,
    operands: tuple[torch.Tensor, ...],
) -> tuple[tuple[torch.Tensor, ...], int | None, int | None]:
    """Cap the explicit ``z`` batch for contraction-path search only."""
    lhs = equation.split("->", 1)[0]
    fragments = [fragment.strip().split() for fragment in lhs.split(",")]
    if len(fragments) != len(operands):
        raise RuntimeError("einsum fragment/operand count mismatch")
    actual_batch = None
    reference_batch = None
    reference_operands = []
    for labels, operand in zip(fragments, operands):
        if "z" not in labels:
            reference_operands.append(operand)
            continue
        axis = labels.index("z")
        operand_batch = int(operand.shape[axis])
        if actual_batch is None:
            actual_batch = operand_batch
            reference_batch = min(operand_batch, _EINSUM_PATH_REFERENCE_BATCH)
        elif operand_batch != actual_batch:
            raise RuntimeError(
                "inconsistent single-site batch dimensions: "
                f"{actual_batch} and {operand_batch}"
            )
        if operand_batch > int(reference_batch):
            operand = operand.narrow(axis, 0, int(reference_batch))
        reference_operands.append(operand)
    return tuple(reference_operands), actual_batch, reference_batch


def _binary_einsum_path(
    equation: str,
    operands: tuple[torch.Tensor, ...],
    *,
    strategy: str,
    memory_limit: int,
) -> list[int] | None:
    optimizer = torch.backends.opt_einsum.get_opt_einsum()
    pairs = optimizer.contract_path(
        equation, *operands, optimize=strategy, memory_limit=memory_limit
    )[0]
    if any(len(pair) != 2 for pair in pairs):
        return None
    return [index for pair in pairs for index in pair]


def _path_fits_memory_limit(
    equation: str,
    operands: tuple[torch.Tensor, ...],
    path: list[int],
    *,
    memory_limit: int,
) -> bool:
    optimizer = torch.backends.opt_einsum.get_opt_einsum()
    pairs = [tuple(path[start : start + 2]) for start in range(0, len(path), 2)]
    _, info = optimizer.contract_path(equation, *operands, optimize=pairs)
    return int(info.largest_intermediate) <= memory_limit


def _einsum(
    equation: str,
    *operands: torch.Tensor,
    memory_limit: int,
) -> torch.Tensor:
    if len(operands) <= 2 or not torch.backends.opt_einsum.enabled:
        return torch.einsum(equation, *operands)
    reference, actual_batch, reference_batch = _reference_batch_operands(
        equation, operands
    )
    strategy = torch.backends.opt_einsum.strategy
    key = (
        equation,
        tuple(tuple(int(x) for x in value.shape) for value in operands),
        strategy,
        memory_limit,
    )
    path = _EINSUM_PATH_CACHE.get(key)
    if path is None:
        reference_key = (
            equation,
            tuple(tuple(int(x) for x in value.shape) for value in reference),
            strategy,
            memory_limit,
        )
        if reference_key not in _EINSUM_REFERENCE_PATH_CACHE:
            _EINSUM_REFERENCE_PATH_CACHE[reference_key] = _binary_einsum_path(
                equation, reference, strategy=strategy, memory_limit=memory_limit
            )
        path = _EINSUM_REFERENCE_PATH_CACHE[reference_key]
        inherited = (
            actual_batch is not None
            and reference_batch is not None
            and actual_batch > reference_batch
        )
        if (
            path is not None
            and inherited
            and not _path_fits_memory_limit(
                equation, operands, path, memory_limit=memory_limit
            )
        ):
            path = None
        if path is None:
            path = _binary_einsum_path(
                equation, operands, strategy=strategy, memory_limit=memory_limit
            )
        if path is None:
            raise RuntimeError(
                "no binary single-site contraction path fits the memory limit"
            )
        _EINSUM_PATH_CACHE[key] = path
    return torch._VF.einsum(equation, operands, path=path)


def _is_readout_path_memory_error(error: RuntimeError) -> bool:
    message = str(error)
    return (
        "no binary single-site contraction path fits the memory limit" in message
        or "readout contraction exceeds einsum memory limit" in message
        or "has no exact connected binary tree" in message
        or "exceeds row-transfer memory limit" in message
    )


def _shared_gate_factors(
    operators: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (
        int(operators.data_ptr()),
        tuple(int(x) for x in operators.shape),
        str(operators.dtype),
        str(operators.device),
    )
    entry = _GATE_FACTOR_CACHE.get(key)
    if entry is None or entry[0] is not operators:
        gate4 = operators.reshape(1, 2, 2, 2, 2)
        matrix = gate4.permute(0, 1, 3, 2, 4).reshape(1, 4, 4)
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
        # Preserve the exact operator-Schmidt rank.  In particular, the
        # identity used to zip a residual MPO bond has rank one; retaining
        # all four formal SVD columns multiplies the dominant PEPS QR
        # intermediate by four and can exceed an A100-80GB even for B=1.
        threshold = (
            torch.finfo(S.dtype).eps
            * max(matrix.shape[-2:])
            * S[..., :1].abs().clamp_min(1.0)
        )
        rank = max(1, int((S > threshold).sum().item()))
        U = U[..., :rank]
        S = S[..., :rank]
        Vh = Vh[..., :rank, :]
        root = torch.sqrt(S)
        left = (U * root.unsqueeze(1)).reshape(1, 2, 2, rank)
        right = (root.unsqueeze(2) * Vh).reshape(1, rank, 2, 2)
        entry = (operators, left, right)
        _GATE_FACTOR_CACHE[key] = entry
    return entry[1], entry[2]


def _randomized_svd(
    matrix: torch.Tensor,
    *,
    max_rank: int,
    oversample: int,
    power_iterations: int,
    random_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Return a deterministic low-rank SVD and exact input norm metadata.

    Only a ``max_rank + oversample`` subspace is materialized.  The returned
    singular vectors are approximate, but the total matrix Frobenius norm is
    evaluated exactly.  Consequently ``_adaptive_svd_rank`` can include all
    unresolved spectral weight when checking the requested tail tolerance.
    Batched matrices are supported and use one reproducible random probe per
    batch member.
    """
    if matrix.ndim < 2:
        raise ValueError("randomized SVD expects a matrix or matrix batch")
    rows, columns = map(int, matrix.shape[-2:])
    available = min(rows, columns)
    target = min(available, int(max_rank) + int(oversample))
    if target < 1:
        raise ValueError("randomized SVD target rank must be positive")
    total_squared_norm = matrix.abs().to(torch.float64).square().sum(dim=(-2, -1))
    if target >= available:
        left, singular, right = torch.linalg.svd(matrix, full_matrices=False)
        return (
            left,
            singular,
            right,
            {
                "available_rank": available,
                "computed_rank": available,
                "spectrum_complete": True,
                "total_squared_norm": total_squared_norm,
                "solver": "full",
            },
        )

    generator = torch.Generator(device=matrix.device)
    generator.manual_seed(int(random_seed))
    probe = torch.randn(
        (*matrix.shape[:-2], columns, target),
        dtype=matrix.dtype,
        device=matrix.device,
        generator=generator,
    )
    sample = matrix @ probe
    del probe
    for _ in range(int(power_iterations)):
        left_basis = torch.linalg.qr(sample, mode="reduced").Q
        right_sample = matrix.mH @ left_basis
        right_basis = torch.linalg.qr(right_sample, mode="reduced").Q
        sample = matrix @ right_basis
        del left_basis, right_sample, right_basis
    basis = torch.linalg.qr(sample, mode="reduced").Q
    del sample
    reduced = basis.mH @ matrix
    small_left, singular, right = torch.linalg.svd(reduced, full_matrices=False)
    left = basis @ small_left
    del basis, reduced, small_left
    return (
        left,
        singular,
        right,
        {
            "available_rank": available,
            "computed_rank": target,
            "spectrum_complete": False,
            "total_squared_norm": total_squared_norm,
            "solver": "randomized",
        },
    )


def _compression_svd(
    matrix: torch.Tensor,
    *,
    max_rank: int,
    solver: str,
    oversample: int,
    power_iterations: int,
    random_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Dispatch a full or memory-bounded compression SVD."""
    if solver == "randomized":
        return _randomized_svd(
            matrix,
            max_rank=max_rank,
            oversample=oversample,
            power_iterations=power_iterations,
            random_seed=random_seed,
        )
    if solver != "full":
        raise ValueError(f"unsupported compression SVD solver {solver!r}")
    left, singular, right = torch.linalg.svd(matrix, full_matrices=False)
    available = int(singular.shape[-1])
    return (
        left,
        singular,
        right,
        {
            "available_rank": available,
            "computed_rank": available,
            "spectrum_complete": True,
            "total_squared_norm": (singular.to(torch.float64).square().sum(dim=-1)),
            "solver": "full",
        },
    )


def _compression_hermitian_eigh(
    matrix: torch.Tensor,
    *,
    max_rank: int,
    solver: str,
    oversample: int,
    power_iterations: int,
    random_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Largest eigenpairs of a Hermitian PSD compression matrix.

    The exact trace is returned as the total Schmidt squared norm, so omitted
    eigenvalues remain part of the adaptive discarded-tail check.
    """
    if matrix.ndim < 2 or int(matrix.shape[-2]) != int(matrix.shape[-1]):
        raise ValueError("Hermitian compression solver expects square matrices")
    available = int(matrix.shape[-1])
    total_squared_norm = (
        matrix.diagonal(dim1=-2, dim2=-1)
        .real.to(torch.float64)
        .sum(dim=-1)
        .clamp_min(0.0)
    )
    target = min(available, int(max_rank) + int(oversample))
    if solver == "full" or target >= available:
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        return (
            eigenvalues.flip(-1).clamp_min(0.0),
            eigenvectors.flip(-1),
            {
                "available_rank": available,
                "computed_rank": available,
                "spectrum_complete": True,
                "total_squared_norm": total_squared_norm,
                "solver": "full_eigh",
            },
        )
    if solver != "randomized":
        raise ValueError(f"unsupported compression eigen solver {solver!r}")
    if target < 1:
        raise ValueError("randomized eigen target rank must be positive")
    generator = torch.Generator(device=matrix.device)
    generator.manual_seed(int(random_seed))
    probe = torch.randn(
        (*matrix.shape[:-2], available, target),
        dtype=matrix.dtype,
        device=matrix.device,
        generator=generator,
    )
    sample = matrix @ probe
    del probe
    for _ in range(int(power_iterations)):
        basis = torch.linalg.qr(sample, mode="reduced").Q
        sample = matrix @ basis
        del basis
    basis = torch.linalg.qr(sample, mode="reduced").Q
    del sample
    reduced = basis.mH @ matrix @ basis
    reduced = (reduced + reduced.mH) * 0.5
    eigenvalues, small_vectors = torch.linalg.eigh(reduced)
    eigenvalues = eigenvalues.flip(-1).clamp_min(0.0)
    small_vectors = small_vectors.flip(-1)
    eigenvectors = basis @ small_vectors
    del basis, reduced, small_vectors
    return (
        eigenvalues,
        eigenvectors,
        {
            "available_rank": available,
            "computed_rank": target,
            "spectrum_complete": False,
            "total_squared_norm": total_squared_norm,
            "solver": "randomized_eigh",
        },
    )


def _adaptive_svd_rank(
    singular: torch.Tensor,
    *,
    chi_max: int,
    relative_error: float = 0.0,
    total_squared_norm: torch.Tensor | None = None,
    available_rank: int | None = None,
    spectrum_complete: bool = True,
    solver: str = "full",
    diagnostics_callback: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Choose one batch-compatible rank from a discarded-weight tolerance.

    ``relative_error`` bounds the local relative Frobenius norm of the
    discarded Schmidt tail.  A zero tolerance preserves the historical hard
    ``chi_max`` rule.  Batched updates use the largest rank required by any
    member so every PEPS tensor retains a common shape.
    """
    tolerance = float(relative_error)
    if not math.isfinite(tolerance) or tolerance < 0.0 or tolerance >= 1.0:
        raise ValueError("relative SVD error must be finite and in [0, 1)")
    values = singular if singular.ndim > 1 else singular.unsqueeze(0)
    computed = int(values.shape[-1])
    available = computed if available_rank is None else int(available_rank)
    if available < computed:
        raise ValueError("available SVD rank cannot be smaller than computed rank")
    hard_rank = min(int(chi_max), available, computed)
    if hard_rank < 1:
        raise ValueError("adaptive SVD requires a positive available rank")

    weights = values.to(torch.float64).square()
    if total_squared_norm is None:
        total = weights.sum(dim=-1, keepdim=True)
    else:
        supplied_total = total_squared_norm.to(
            dtype=torch.float64, device=values.device
        )
        if supplied_total.ndim == 0:
            supplied_total = supplied_total.expand(int(values.shape[0]))
        total = supplied_total.reshape(-1, 1)
        if int(total.shape[0]) != int(values.shape[0]):
            raise ValueError("total SVD norm batch does not match singular values")
    retained = weights.cumsum(dim=-1)
    relative_tail = torch.sqrt(
        (
            (total - retained).clamp_min(0.0)
            / total.clamp_min(torch.finfo(weights.dtype).tiny)
        )
    )
    relative_tail = torch.where(
        total > 0.0, relative_tail, torch.zeros_like(relative_tail)
    )

    if tolerance == 0.0:
        rank = hard_rank
        required = torch.full(
            (int(values.shape[0]),), available, dtype=torch.long, device=values.device
        )
    else:
        meets = relative_tail <= tolerance
        first = torch.argmax(meets.to(torch.int64), dim=-1) + 1
        required = torch.where(
            meets.any(dim=-1),
            first,
            torch.full_like(
                first,
                available if spectrum_complete else min(available, computed + 1),
            ),
        )
        rank = min(hard_rank, max(1, int(required.max().item())))

    if diagnostics_callback is not None:
        achieved = relative_tail[:, rank - 1]
        diagnostics_callback(
            {
                "batch_size": int(values.shape[0]),
                "available_rank": available,
                "computed_rank": computed,
                "spectrum_complete": bool(spectrum_complete),
                "svd_solver": str(solver),
                "hard_rank": hard_rank,
                "selected_rank": rank,
                "required_rank_max": int(required.max().item()),
                "requested_relative_error": tolerance,
                "achieved_relative_error_max": float(achieved.max().item()),
                "achieved_relative_error_mean": float(achieved.mean().item()),
                "cap_limited_batches": int((required > hard_rank).sum().item()),
                "required_rank_is_lower_bound": bool(
                    not spectrum_complete and bool((required > computed).any().item())
                ),
            }
        )
    return rank


def _apply_operator_qr(
    tensors: dict[int, torch.Tensor],
    support: tuple[int, ...],
    operators: torch.Tensor,
    *,
    chi_max: int,
    neighbors: dict[int, dict[str, int]],
    cache_shared_gate: bool = True,
    svd_relative_error: float = 0.0,
    svd_solver: str = "full",
    svd_oversample: int = 16,
    svd_power_iterations: int = 2,
    svd_random_seed: int = 0,
    svd_diagnostics_callback: (Callable[[dict[str, Any]], None] | None) = None,
) -> None:
    """Apply a batched 1Q/2Q operator with rank-first QR-SVD reconstruction."""
    if len(support) == 1:
        q = support[0]
        B = int(tensors[q].shape[0])
        op = operators.reshape(-1, 2, 2)
        if op.shape[0] == 1:
            op = op.expand(B, 2, 2)
        if op.shape[0] != B:
            raise ValueError("operator batch does not match PEPS batch")
        value = torch.einsum("bAB,bUDLRB->bAUDLR", op, tensors[q])
        tensors[q] = value.permute(0, 2, 3, 4, 5, 1).contiguous()
        return

    if len(support) != 2:
        raise ValueError(f"unsupported support size {len(support)}")
    q1, q2 = support
    direction = next(
        (side for side, other in neighbors[q1].items() if other == q2), None
    )
    if direction is None:
        raise ValueError(f"support {support} is not nearest-neighbour")
    reverse = OPPOSITE[direction]
    bond1, bond2 = DIR_TO_AXIS[direction], DIR_TO_AXIS[reverse]
    T1, T2 = tensors[q1], tensors[q2]
    B = int(T1.shape[0])

    order1 = [axis for axis in range(4) if axis != bond1] + [bond1, 4]
    order2 = [axis for axis in range(4) if axis != bond2]
    order2.insert(2, bond2)
    order2.append(4)
    perm1 = [0] + [axis + 1 for axis in order1]
    perm2 = [0] + [axis + 1 for axis in order2]
    T1p = T1.permute(*perm1).contiguous()
    T2p = T2.permute(*perm2).contiguous()
    del T1, T2
    left_shape = (T1p.shape[1], T1p.shape[2], T1p.shape[3], 2)
    right_shape = (T2p.shape[1], T2p.shape[2], T2p.shape[4], 2)
    left_size = math.prod(left_shape)
    right_size = math.prod(right_shape)

    gate4 = operators.reshape(-1, 2, 2, 2, 2)
    if gate4.shape[0] not in (1, B):
        raise ValueError("operator batch does not match PEPS batch")
    if gate4.shape[0] == 1 and cache_shared_gate:
        gate_left, gate_right = _shared_gate_factors(operators)
        gate_rank = int(gate_left.shape[-1])
        gate_left = gate_left.expand(B, 2, 2, gate_rank)
        gate_right = gate_right.expand(B, gate_rank, 2, 2)
    else:
        gate_matrix = gate4.permute(0, 1, 3, 2, 4).reshape(B, 4, 4)
        gate_U, gate_S, gate_Vh = torch.linalg.svd(gate_matrix, full_matrices=False)
        root = torch.sqrt(gate_S)
        gate_left = (gate_U * root.unsqueeze(1)).reshape(B, 2, 2, 4)
        gate_right = (root.unsqueeze(2) * gate_Vh).reshape(B, 4, 2, 2)

    left_factor = torch.einsum("Budlxp,BXpg->BudlXxg", T1p, gate_left).reshape(
        B, left_size, -1
    )
    del T1p
    Q_left, R_left = torch.linalg.qr(left_factor, mode="reduced")
    del left_factor
    right_factor = torch.einsum("BgYq,Babxrq->BxgabrY", gate_right, T2p).reshape(
        B, -1, right_size
    )
    del T2p
    Q_right, R_right = torch.linalg.qr(right_factor.transpose(-2, -1), mode="reduced")
    del right_factor
    core = R_left @ R_right.transpose(-2, -1)
    del R_left, R_right
    try:
        core_U, singular, core_Vh, svd_metadata = _compression_svd(
            core,
            max_rank=int(chi_max),
            solver=str(svd_solver),
            oversample=int(svd_oversample),
            power_iterations=int(svd_power_iterations),
            random_seed=int(svd_random_seed),
        )
    except torch._C._LinAlgError:
        # CUDA's Jacobi SVD can fail on the highly repeated singular values
        # produced by an exact SWAP/identity zip after a residual-MPO update.
        # The core is at most O(chi) rather than a PEPS-sized tensor, so a
        # complex128 CPU fallback is inexpensive and avoids changing either
        # the operator or the requested PEPS truncation.
        if bool((~torch.isfinite(core)).any()):
            raise RuntimeError("non-finite two-site QR core before SVD")
        core_cpu = core.detach().to(device="cpu")
        cpu_U, cpu_singular, cpu_Vh = torch.linalg.svd(core_cpu, full_matrices=False)
        core_U = cpu_U.to(device=core.device)
        singular = cpu_singular.to(device=core.device)
        core_Vh = cpu_Vh.to(device=core.device)
        svd_metadata = {
            "available_rank": int(singular.shape[-1]),
            "computed_rank": int(singular.shape[-1]),
            "spectrum_complete": True,
            "total_squared_norm": (singular.to(torch.float64).square().sum(dim=-1)),
            "solver": "cpu_full_fallback",
        }
        del core_cpu, cpu_U, cpu_singular, cpu_Vh
    del core
    # Decide rank before expanding the small-core singular vectors.  The old
    # order kept full U/Vh, both expanded gate factors and both Q matrices
    # alive while allocating the truncated results (a D=96 OOM hotspot).
    rank = _adaptive_svd_rank(
        singular,
        chi_max=int(chi_max),
        relative_error=float(svd_relative_error),
        total_squared_norm=svd_metadata["total_squared_norm"],
        available_rank=int(svd_metadata["available_rank"]),
        spectrum_complete=bool(svd_metadata["spectrum_complete"]),
        solver=str(svd_metadata["solver"]),
        diagnostics_callback=svd_diagnostics_callback,
    )
    root = torch.sqrt(singular[:, :rank])
    del singular, svd_metadata
    inverse1 = [0] * 6
    inverse2 = [0] * 6
    for destination, source in enumerate(perm1):
        inverse1[source] = destination
    for destination, source in enumerate(perm2):
        inverse2[source] = destination

    # Scale only the small retained core, then restore/release one side before
    # constructing the other.  This changes storage lifetime, not truncation.
    left_core = core_U[:, :, :rank] * root.unsqueeze(1)
    del core_U
    left = Q_left @ left_core
    del Q_left, left_core
    new_left = left.reshape(B, *left_shape, rank).permute(0, 1, 2, 3, 5, 4)
    tensors[q1] = new_left.permute(*inverse1).contiguous()
    del left, new_left

    right_core = root.unsqueeze(2) * core_Vh[:, :rank, :]
    del core_Vh, root
    right = right_core @ Q_right.transpose(-2, -1)
    del Q_right, right_core
    new_right = right.reshape(B, rank, *right_shape).permute(0, 2, 3, 1, 4, 5)
    tensors[q2] = new_right.permute(*inverse2).contiguous()


def _apply_operator_microbatched(
    tensors: dict[int, torch.Tensor],
    support: tuple[int, ...],
    operators: torch.Tensor,
    *,
    config: SingleSiteConfig,
    neighbors: dict[int, dict[str, int]],
    svd_relative_error: float = 0.0,
    svd_diagnostics_callback: (Callable[[dict[str, Any]], None] | None) = None,
) -> None:
    if len(support) == 1:
        _apply_operator_qr(
            tensors,
            support,
            operators,
            chi_max=config.chi_max,
            neighbors=neighbors,
            cache_shared_gate=(operators.shape[0] == 1),
            svd_relative_error=svd_relative_error,
            svd_solver=config.svd_solver,
            svd_oversample=config.svd_oversample,
            svd_power_iterations=config.svd_power_iterations,
            svd_random_seed=config.svd_random_seed,
            svd_diagnostics_callback=svd_diagnostics_callback,
        )
        return
    B = int(next(iter(tensors.values())).shape[0])
    step = min(config.two_qubit_apply_batch_size, B)
    operator_batch = operators.reshape(-1, 2, 2, 2, 2)
    shared = operator_batch.shape[0] == 1
    q1, q2 = support
    left_result = None
    right_result = None
    for start in range(0, B, step):
        stop = min(start + step, B)
        endpoints = {q1: tensors[q1][start:stop], q2: tensors[q2][start:stop]}
        local_operator = operators if shared else operator_batch[start:stop]
        _apply_operator_qr(
            endpoints,
            support,
            local_operator,
            chi_max=config.chi_max,
            neighbors=neighbors,
            cache_shared_gate=shared,
            svd_relative_error=svd_relative_error,
            svd_solver=config.svd_solver,
            svd_oversample=config.svd_oversample,
            svd_power_iterations=config.svd_power_iterations,
            svd_random_seed=config.svd_random_seed,
            svd_diagnostics_callback=svd_diagnostics_callback,
        )
        left_part = endpoints[q1]
        right_part = endpoints[q2]
        if left_result is None:
            left_result = torch.empty(
                (B, *left_part.shape[1:]),
                dtype=left_part.dtype,
                device=left_part.device,
            )
            right_result = torch.empty(
                (B, *right_part.shape[1:]),
                dtype=right_part.dtype,
                device=right_part.device,
            )
        elif tuple(left_part.shape[1:]) != tuple(left_result.shape[1:]) or tuple(
            right_part.shape[1:]
        ) != tuple(right_result.shape[1:]):
            raise RuntimeError(
                "two-qubit microbatches produced inconsistent output shapes"
            )
        left_result[start:stop].copy_(left_part)
        right_result[start:stop].copy_(right_part)
    if left_result is None or right_result is None:
        raise RuntimeError("empty two-qubit operator batch")
    tensors[q1] = left_result
    tensors[q2] = right_result


def _apply_disjoint_2q_layer_batched(
    tensors: dict[int, torch.Tensor],
    gates: list[dict[str, Any]],
    *,
    config: SingleSiteConfig,
    neighbors: dict[int, dict[str, int]],
) -> set[tuple[int, str]]:
    """Apply compatible disjoint 2Q gates with one gate/branch batch axis.

    Gates are grouped by orientation and endpoint tensor shapes.  Concatenating
    the gate and branch axes is exact because every support in the layer is
    disjoint; no tensor written by one QR--SVD is read by another gate.
    """
    B = int(next(iter(tensors.values())).shape[0])
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    invalidated: set[tuple[int, str]] = set()
    for gate in gates:
        support = tuple(int(q) for q in gate["qubits"])
        if len(support) != 2:
            raise ValueError("disjoint 2Q layer batching received a non-2Q gate")
        q1, q2 = support
        direction = next(
            (side for side, other in neighbors[q1].items() if other == q2),
            None,
        )
        if direction is None:
            raise ValueError(f"support {support} is not nearest-neighbour")
        key = (
            direction,
            tuple(int(x) for x in tensors[q1].shape[1:]),
            tuple(int(x) for x in tensors[q2].shape[1:]),
        )
        buckets.setdefault(key, []).append(gate)
        invalidated.update(
            {
                (q1, direction),
                (q2, OPPOSITE[direction]),
            }
        )

    gate_step = int(config.two_qubit_layer_gate_batch_size)
    for (direction, _, _), bucket in buckets.items():
        for gate_start in range(0, len(bucket), gate_step):
            selected = bucket[gate_start : gate_start + gate_step]
            left_sites = [int(gate["qubits"][0]) for gate in selected]
            right_sites = [int(gate["qubits"][1]) for gate in selected]
            synthetic = {
                0: torch.cat([tensors[site] for site in left_sites], dim=0),
                1: torch.cat([tensors[site] for site in right_sites], dim=0),
            }
            operator_parts = []
            for gate in selected:
                operator = gate["ideal_unitary"].reshape(-1, 4, 4)
                if int(operator.shape[0]) == 1:
                    operator = operator.expand(B, 4, 4)
                elif int(operator.shape[0]) != B:
                    raise ValueError(
                        "layer gate operator batch does not match PEPS batch"
                    )
                operator_parts.append(operator)
            operators = torch.cat(operator_parts, dim=0)
            synthetic_neighbors = {
                0: {direction: 1},
                1: {OPPOSITE[direction]: 0},
            }
            _apply_operator_microbatched(
                synthetic,
                (0, 1),
                operators,
                config=config,
                neighbors=synthetic_neighbors,
            )
            left_parts = synthetic[0].split(B, dim=0)
            right_parts = synthetic[1].split(B, dim=0)
            for site, value in zip(left_sites, left_parts):
                tensors[site] = value.clone()
            for site, value in zip(right_sites, right_parts):
                tensors[site] = value.clone()
    return invalidated


def _apply_layer(
    tensors: dict[int, torch.Tensor],
    gates: list[dict[str, Any]],
    *,
    config: SingleSiteConfig,
    neighbors: dict[int, dict[str, int]],
) -> set[tuple[int, str]]:
    """Apply a layer and return message keys invalidated by two-site SVDs.

    A two-qubit QR/SVD changes the ket/bra coordinate system on the gate
    bond.  Messages on that bond may still have the same shape while being
    expressed in the old basis, so they cannot be used as warm starts.
    Messages on all other bonds retain their coordinate systems.
    """
    supports = [tuple(int(q) for q in gate["qubits"]) for gate in gates]
    two_qubit_disjoint = (
        bool(supports)
        and all(len(support) == 2 for support in supports)
        and len({site for support in supports for site in support}) == 2 * len(supports)
    )
    if config.batch_disjoint_two_qubit_gates and two_qubit_disjoint:
        return _apply_disjoint_2q_layer_batched(
            tensors, gates, config=config, neighbors=neighbors
        )

    invalidated: set[tuple[int, str]] = set()
    for gate in gates:
        support = tuple(int(q) for q in gate["qubits"])
        _apply_operator_microbatched(
            tensors, support, gate["ideal_unitary"], config=config, neighbors=neighbors
        )
        if len(support) == 2:
            q1, q2 = support
            direction = next(
                (side for side, other in neighbors[q1].items() if other == q2),
                None,
            )
            if direction is None:
                raise ValueError(f"support {support} is not nearest-neighbour")
            invalidated.update(
                {
                    (q1, direction),
                    (q2, OPPOSITE[direction]),
                }
            )
    return invalidated


def _is_one_qubit_layer(gates: list[dict[str, Any]]) -> bool:
    return bool(gates) and all(len(gate["qubits"]) == 1 for gate in gates)


def _build_plan(ctx: Any, config: SingleSiteConfig) -> dict[str, Any]:
    neighbors = _build_neighbors(int(ctx.width), int(ctx.length))
    directed_edges = tuple(
        (site, direction)
        for site in sorted(neighbors)
        for direction in "UDLR"
        if direction in neighbors[site]
    )
    seen = set()
    undirected_edges = []
    for site, direction in directed_edges:
        other = neighbors[site][direction]
        edge = (min(site, other), max(site, other))
        if edge not in seen:
            seen.add(edge)
            undirected_edges.append(edge)
    targets = {}
    last_partner: dict[int, int] = {}
    for layer, gates in enumerate(ctx.layers):
        supports = {
            int(gate["gate_idx"]): tuple(int(q) for q in gate["qubits"])
            for gate in gates
        }
        readout_supports = {}
        for gate_id, support in supports.items():
            # The readout shape is a source-level configuration.  ``strip``
            # and ``2x2`` therefore give a 1Q target a neighbouring anchor
            # exactly as they do in the established source1 2x2 policy;
            # target arity changes only which physical legs remain open.
            if config.readout_block == "support" or len(support) == 2:
                readout_supports[gate_id] = support
            elif support[0] in last_partner:
                readout_supports[gate_id] = (support[0], last_partner[support[0]])
            else:
                site = support[0]
                partner = next(
                    (
                        neighbors[site][direction]
                        for direction in "RDLU"
                        if direction in neighbors[site]
                    ),
                    None,
                )
                readout_supports[gate_id] = (
                    support if partner is None else (site, partner)
                )
        targets[layer] = {
            "gate_ids": tuple(int(gate["gate_idx"]) for gate in gates),
            "gate_to_local": {
                int(gate["gate_idx"]): local for local, gate in enumerate(gates)
            },
            "supports": supports,
            "readout_supports": readout_supports,
        }
        for support in supports.values():
            if len(support) == 2:
                first, second = support
                last_partner[first] = second
                last_partner[second] = first
    element_size = torch.empty((), dtype=ctx.dtype).element_size()
    return {
        "width": int(ctx.width),
        "length": int(ctx.length),
        "neighbors": neighbors,
        "directed_edges": directed_edges,
        "undirected_edges": tuple(undirected_edges),
        "targets": targets,
        "local_readout_method": config.local_readout_method,
        "gloop_size": config.gloop_size,
        "gloop_combine": config.gloop_combine,
        "gloop_grow_from": config.gloop_grow_from,
        "gloop_optimize": config.gloop_optimize,
        "gloop_expression_cache_size": config.gloop_expression_cache_size,
        "gloop_memory_target_gib": config.gloop_memory_target_gib,
        "gloop_memory_search_repeats": (config.gloop_memory_search_repeats),
        "gloop_memory_max_slices": config.gloop_memory_max_slices,
        "gloop_peak_budget_gib": config.gloop_peak_budget_gib,
        "gloop_peak_reserve_gib": config.gloop_peak_reserve_gib,
        "gloop_peak_safety_factor": config.gloop_peak_safety_factor,
        "gloop_oom_max_retries": config.gloop_oom_max_retries,
        "gloop_memory_optimizer": None,
        "gloop_audit": {
            "calls": 0,
            "region_contractions": 0,
            "expression_cache_hits": 0,
            "expression_cache_misses": 0,
            "expression_cache_evictions": 0,
            "expression_build_seconds": 0.0,
            "max_expression_build_seconds": 0.0,
            "oom_retry_count": 0,
            "oom_retry_min_branch_chunk": None,
            "max_region_sites": 0,
            "region_count_by_size": {},
            "memory_planned_expression_count": 0,
            "memory_researched_expression_count": 0,
            "memory_sliced_expression_count": 0,
            "memory_max_original_intermediate_elements": 0,
            "memory_max_final_intermediate_elements": 0,
            "memory_max_slice_count": 1,
        },
        "readout_block": config.readout_block,
        "readout_boundary_mode": config.readout_boundary_mode,
        "readout_contraction_mode": config.readout_contraction_mode,
        "joint_halo_rank": config.joint_halo_rank,
        "joint_halo_oversample": config.joint_halo_oversample,
        "joint_halo_power_iterations": config.joint_halo_power_iterations,
        "joint_halo_random_seed": config.joint_halo_random_seed,
        "joint_halo_rank_audit": {
            "factorization_count": 0,
            "exact_factor_bypass_count": 0,
            "effective_rank_min": None,
            "effective_rank_max": None,
            "max_probe_relative_residual": 0.0,
        },
        "bounded_2x2_path_audit": {
            "fallback_contraction_count": 0,
            "canonical_tree_contraction_count": 0,
            "max_planned_intermediate_elements": 0,
            "canonical_memory_cache_hits": 0,
            "canonical_builtin_cache_hits": 0,
            "canonical_disk_cache_hits": 0,
            "canonical_plans_created": 0,
            "canonical_dp_wins": 0,
            "canonical_greedy_wins": 0,
            "canonical_greedy_only_plans": 0,
            "canonical_planning_seconds": 0.0,
            "persistent_entries_queued": 0,
            "persistent_cache_entries_loaded": 0,
            "persistent_cache_load_errors": 0,
            "persistent_cache_validation_errors": 0,
        },
        "batch_readout_patches": config.batch_readout_patches,
        "readout_patch_batch_size": config.readout_patch_batch_size,
        "batch_bethe_sites": config.batch_bethe_sites,
        "bethe_site_batch_size": config.bethe_site_batch_size,
        "auto_readout_tiling": config.auto_readout_tiling,
        "fuse_readout_operators": config.fuse_readout_operators,
        "reuse_one_qubit_readout": config.reuse_one_qubit_readout,
        "memory_limit": max(1, int(config.einsum_memory_gib * 1024**3 / element_size)),
    }


def _build_one_qubit_readout_reuse_plan(
    ctx: Any,
    clean_cache: dict[str, Any],
    parent_layer: int,
) -> dict[str, Any] | None:
    """Plan exact next-layer 1Q projections on a preceding 2Q readout.

    If ``rho`` is the mixed local transition before a one-qubit ideal unitary
    ``U``, the next-layer target insertion obeys

        Tr(E U rho U_dag) = Tr((U_dag E U) rho).

    The pulled-back operator is embedded on the matching endpoint of the
    preceding two-qubit support.  Only sites whose configured 2x2 readout uses
    that exact preceding bond are covered; all other sites retain the ordinary
    readout path.
    """
    config = clean_cache["config"]
    plan = clean_cache["plan"]
    next_layer = int(parent_layer) + 1
    if (
        not config.reuse_one_qubit_readout
        or clean_cache.get("background_layer_hook") is not None
        or next_layer >= len(ctx.layers)
        or not config.fuse_readout_operators
        or config.readout_block != "2x2"
        or config.branch_weight_mode != "source_anchored"
        or config.target_amplitude_filter_tol != 0.0
        or config.auto_quarantine_ill_conditioned
        or config.record_support_consistency
        or config.retain_quarantine_ledger
        or not ctx.layers[parent_layer]
        or not all(len(gate["qubits"]) == 2 for gate in ctx.layers[parent_layer])
        or not _is_one_qubit_layer(ctx.layers[next_layer])
    ):
        return None

    parent_target = plan["targets"][parent_layer]
    next_target = plan["targets"][next_layer]
    parent_by_bond: dict[
        frozenset[int],
        tuple[
            tuple[int, ...],
            tuple[int, ...],
        ],
    ] = {}
    for gate_id in parent_target["gate_ids"]:
        support = tuple(parent_target["supports"][gate_id])
        readout_support = tuple(parent_target["readout_supports"][gate_id])
        if len(support) != 2 or len(readout_support) != 2:
            continue
        parent_by_bond[frozenset(support)] = (support, readout_support)

    by_cache_key: dict[tuple, list[dict[str, Any]]] = {}
    covered_gate_ids = []
    dressed = clean_cache["dressed"][next_layer]
    for gate in ctx.layers[next_layer]:
        gate_id = int(gate["gate_idx"])
        support = tuple(next_target["supports"][gate_id])
        readout_support = tuple(next_target["readout_supports"][gate_id])
        if len(support) != 1 or len(readout_support) != 2:
            continue
        parent = parent_by_bond.get(frozenset(readout_support))
        if parent is None:
            continue
        parent_support, parent_readout_support = parent
        if support[0] not in parent_support:
            continue
        if _select_readout_patch(parent_readout_support, plan) != _select_readout_patch(
            readout_support, plan
        ):
            continue

        unitary = gate["ideal_unitary"].reshape(2, 2)
        modes = dressed[gate_id].reshape(-1, 2, 2)
        pulled = torch.einsum(
            "ab,vbc,cd->vad",
            unitary.conj().transpose(-1, -2),
            modes,
            unitary,
        )
        identity = torch.eye(2, dtype=pulled.dtype, device=pulled.device)
        if support[0] == parent_support[0]:
            embedded = torch.einsum("vab,cd->vacbd", pulled, identity).reshape(-1, 4, 4)
        else:
            embedded = torch.einsum("ab,vcd->vacbd", identity, pulled).reshape(-1, 4, 4)
        cache_key = (parent_support, parent_readout_support)
        by_cache_key.setdefault(cache_key, []).append(
            {
                "gate_id": gate_id,
                "target_local": int(next_target["gate_to_local"][gate_id]),
                "operators": embedded,
            }
        )
        covered_gate_ids.append(gate_id)

    if not covered_gate_ids:
        return None
    return {
        "parent_layer": int(parent_layer),
        "target_layer": next_layer,
        "target_gate_count": len(next_target["gate_ids"]),
        "covered_gate_ids": tuple(covered_gate_ids),
        "by_cache_key": {key: tuple(entries) for key, entries in by_cache_key.items()},
    }


def _edge_shape(
    ket: dict[int, torch.Tensor],
    bra: dict[int, torch.Tensor],
    site: int,
    direction: str,
) -> tuple[int, int]:
    axis = DIR_TO_AXIS[direction]
    return int(ket[site].shape[axis + 1]), int(bra[site].shape[axis])


def _normalize_message(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B = int(value.shape[0])
    norm = torch.linalg.norm(value.reshape(B, -1), dim=1)
    bad = (~torch.isfinite(norm)) | (norm <= 1e-30)
    safe = torch.where(bad, torch.ones_like(norm), norm)
    shape = (B,) + (1,) * (value.ndim - 1)
    return value / safe.reshape(shape), bad, norm


def _initialize_messages(
    ket: dict[int, torch.Tensor],
    bra: dict[int, torch.Tensor],
    plan: dict[str, Any],
    *,
    warm: dict[tuple[int, str], torch.Tensor] | None,
    clean: dict[tuple[int, str], torch.Tensor] | None,
    invalidated: set[tuple[int, str]] | None = None,
) -> tuple[dict[tuple[int, str], torch.Tensor], str]:
    """Warm-start each edge independently, falling back only where needed."""
    B = int(next(iter(ket.values())).shape[0])
    _emit_stage_profile_event(
        "bp_initialization_warm_start",
        "begin",
        batch_rows=B,
        has_warm=warm is not None,
        has_clean=clean is not None,
    )
    ref = next(iter(ket.values()))
    expected = {
        key: (B, *_edge_shape(ket, bra, key[0], key[1]))
        for key in plan["directed_edges"]
    }
    messages = {}
    origins = set()
    invalidated = invalidated or set()
    for key, shape in expected.items():
        value = None
        origin = "cold"
        for candidate, label in ((warm, "previous_mixed"), (clean, "clean")):
            if candidate is None:
                continue
            if key in invalidated:
                continue
            candidate_value = candidate.get(key)
            if (
                candidate_value is None
                or candidate_value.dtype != ref.dtype
                or candidate_value.device != ref.device
            ):
                continue
            if tuple(candidate_value.shape) == shape:
                value = candidate_value.clone()
            elif (
                candidate_value.ndim == 2 and tuple(candidate_value.shape) == shape[1:]
            ):
                value = candidate_value.unsqueeze(0).expand(shape).clone()
            elif (
                candidate_value.ndim == 3
                and candidate_value.shape[0] == 1
                and tuple(candidate_value.shape[1:]) == shape[1:]
            ):
                value = candidate_value.expand(shape).clone()
            if value is not None:
                origin = label
                break
        if value is None:
            _, d_ket, d_bra = shape
            value = (
                torch.eye(
                    d_ket,
                    d_bra,
                    dtype=ref.dtype,
                    device=ref.device,
                )
                .unsqueeze(0)
                .expand(shape)
                .clone()
            )
        value, bad, _ = _normalize_message(value)
        if bool(bad.any()):
            raise RuntimeError(f"message {key}: zero/non-finite norm")
        messages[key] = value
        origins.add(origin)
    labels = tuple(
        label for label in ("previous_mixed", "clean", "cold") if label in origins
    )
    result_label = labels[0] if len(labels) == 1 else "partial_" + "+".join(labels)
    _emit_stage_profile_event(
        "bp_initialization_warm_start", "end", batch_rows=B, init_label=result_label
    )
    return messages, result_label


def _contract_site(
    ket: dict[int, torch.Tensor],
    bra: dict[int, torch.Tensor],
    site: int,
    messages: dict[tuple[int, str], torch.Tensor],
    plan: dict[str, Any],
    *,
    out_dir: str | None = None,
    keep_phys: bool = False,
) -> torch.Tensor:
    """Contract one mixed double-layer site with its incoming messages."""
    bra_value = bra[site].conj()
    batched_bra = bra_value.ndim == ket[site].ndim
    if batched_bra and int(bra_value.shape[0]) != int(ket[site].shape[0]):
        raise ValueError("batched bra does not match the ket batch")
    fragments = [
        "z u d l r p",
        ("z " if batched_bra else "") + "U D L R " + ("P" if keep_phys else "p"),
    ]
    operands = [ket[site], bra_value]
    for direction in "UDLR":
        if direction == out_dir or direction not in plan["neighbors"][site]:
            continue
        other = plan["neighbors"][site][direction]
        fragments.append("z " + direction.lower() + " " + direction.upper())
        operands.append(messages[(other, OPPOSITE[direction])])
    output = ["z"]
    if keep_phys:
        output.extend(("p", "P"))
    if out_dir is not None:
        output.extend((out_dir.lower(), out_dir.upper()))
    equation = ", ".join(fragments) + " -> " + " ".join(output)
    return _einsum(equation, *operands, memory_limit=plan["memory_limit"])


def _bp_edge_plan(plan: dict[str, Any]) -> tuple:
    """Compile the fixed single-site BP equations once per lattice plan."""
    cached = plan.get("bp_edge_plan")
    if cached is not None:
        return cached
    entries = []
    for site, out_dir in plan["directed_edges"]:
        fragments = ["z u d l r p", "U D L R p"]
        incoming = []
        for direction in "UDLR":
            if direction == out_dir or direction not in plan["neighbors"][site]:
                continue
            other = plan["neighbors"][site][direction]
            incoming.append((other, OPPOSITE[direction]))
            fragments.append("z " + direction.lower() + " " + direction.upper())
        equation = (
            ", ".join(fragments) + " -> z " + out_dir.lower() + " " + out_dir.upper()
        )
        entries.append((site, out_dir, equation, tuple(incoming)))
    compiled = tuple(entries)
    plan["bp_edge_plan"] = compiled
    return compiled


def _bp_site_plan(plan: dict[str, Any]) -> tuple:
    """Group the directed-edge plan without changing its site/edge order."""
    cached = plan.get("bp_site_plan")
    if cached is not None:
        return cached
    grouped = []
    current_site = None
    current_entries = []
    for entry in _bp_edge_plan(plan):
        site = entry[0]
        if current_site is not None and site != current_site:
            grouped.append((current_site, tuple(current_entries)))
            current_entries = []
        current_site = site
        current_entries.append(entry)
    if current_site is not None:
        grouped.append((current_site, tuple(current_entries)))
    compiled = tuple(grouped)
    plan["bp_site_plan"] = compiled
    return compiled


def _contract_bp_edge(
    ket: dict[int, torch.Tensor],
    bra_conj: dict[int, torch.Tensor],
    messages: dict[tuple[int, str], torch.Tensor],
    site: int,
    equation: str,
    incoming: tuple[tuple[int, str], ...],
    plan: dict[str, Any],
) -> torch.Tensor:
    operands = [ket[site], bra_conj[site]]
    operands.extend(messages[key] for key in incoming)
    return _einsum(equation, *operands, memory_limit=plan["memory_limit"])


def _contract_bp_opposite_pair(
    ket: dict[int, torch.Tensor],
    bra_conj: dict[int, torch.Tensor],
    messages: dict[tuple[int, str], torch.Tensor],
    site: int,
    pair: tuple[str, str],
    plan: dict[str, Any],
    *,
    batched_bra: bool = False,
) -> dict[tuple[int, str], torch.Tensor] | None:
    """Share one cavity contraction for two opposite outgoing messages.

    The common tensor leaves both opposite doubled legs open, then contracts
    the incoming message on the other leg to obtain each cavity.  If that
    common tensor would exceed the configured einsum workspace, return
    ``None`` and let the caller use the historical per-edge contractions.
    """
    first, second = pair
    site_neighbors = plan["neighbors"][site]
    if first not in site_neighbors or second not in site_neighbors:
        return None
    B = int(ket[site].shape[0])
    common_elements = B
    for direction in pair:
        axis = DIR_TO_AXIS[direction]
        common_elements *= int(ket[site].shape[axis + 1])
        common_elements *= int(bra_conj[site].shape[axis + int(batched_bra)])
    if common_elements > int(plan["memory_limit"]):
        return None

    fragments = ["z u d l r p", "z U D L R p" if batched_bra else "U D L R p"]
    operands = [ket[site], bra_conj[site]]
    for direction in "UDLR":
        if direction in pair or direction not in site_neighbors:
            continue
        other = site_neighbors[direction]
        fragments.append("z " + direction.lower() + " " + direction.upper())
        operands.append(messages[(other, OPPOSITE[direction])])
    common_output = ["z"]
    for direction in pair:
        common_output.extend((direction.lower(), direction.upper()))
    common_equation = ", ".join(fragments) + " -> " + " ".join(common_output)
    common = _einsum(common_equation, *operands, memory_limit=plan["memory_limit"])

    result = {}
    for out_direction, incoming_direction in (
        (first, second),
        (second, first),
    ):
        other = site_neighbors[incoming_direction]
        incoming = messages[(other, OPPOSITE[incoming_direction])]
        equation = (
            " ".join(common_output)
            + ", z "
            + incoming_direction.lower()
            + " "
            + incoming_direction.upper()
            + " -> z "
            + out_direction.lower()
            + " "
            + out_direction.upper()
        )
        result[(site, out_direction)] = _einsum(
            equation, common, incoming, memory_limit=plan["memory_limit"]
        )
    return result


def _contract_bp_site_messages(
    ket: dict[int, torch.Tensor],
    bra_conj: dict[int, torch.Tensor],
    messages: dict[tuple[int, str], torch.Tensor],
    site: int,
    entries: tuple,
    plan: dict[str, Any],
    *,
    reuse_opposite_cavities: bool,
) -> dict[tuple[int, str], torch.Tensor]:
    """Evaluate one site's outgoing map while preserving Gauss--Seidel order."""
    raw_by_key: dict[tuple[int, str], torch.Tensor] = {}
    if reuse_opposite_cavities:
        for pair in (("U", "D"), ("L", "R")):
            shared = _contract_bp_opposite_pair(
                ket, bra_conj, messages, site, pair, plan
            )
            if shared is not None:
                raw_by_key.update(shared)
    for _, direction, equation, incoming in entries:
        key = (site, direction)
        if key not in raw_by_key:
            raw_by_key[key] = _contract_bp_edge(
                ket, bra_conj, messages, site, equation, incoming, plan
            )
    return raw_by_key


def _phase_align(raw: torch.Tensor, old: torch.Tensor) -> torch.Tensor:
    B = int(raw.shape[0])
    overlap = torch.linalg.vecdot(old.reshape(B, -1), raw.reshape(B, -1), dim=1)
    flat = raw.reshape(B, -1)
    pivot = flat.gather(1, flat.abs().argmax(dim=1, keepdim=True)).squeeze(1)
    reference = torch.where(overlap.abs() > 1e-12, overlap, pivot)
    phase = torch.where(
        reference.abs() > 1e-12,
        torch.angle(reference),
        torch.zeros_like(reference.real),
    )
    return raw * torch.exp(-1j * phase).reshape(B, 1, 1)


def _projective_residual(raw: torch.Tensor, old: torch.Tensor) -> torch.Tensor:
    """Return a stable phase-invariant distance between normalized messages.

    Both arguments are Frobenius-normalized by the caller.  The raw overlap
    formula ``sqrt(1 - |overlap|**2)`` loses precision when the messages are
    already close, so phase-align first and measure the Frobenius distance.
    This remains independent of the arbitrary message phase without a
    near-one subtraction.
    """
    B = int(raw.shape[0])
    overlap = torch.linalg.vecdot(old.reshape(B, -1), raw.reshape(B, -1), dim=1)
    magnitude = overlap.abs().clamp_min(1e-30)
    phase = torch.where(
        overlap.abs() > 1e-12,
        overlap.conj() / magnitude,
        torch.ones_like(overlap),
    )
    aligned = raw * phase.reshape(B, 1, 1)
    return torch.linalg.vector_norm((aligned - old).reshape(B, -1), dim=1)


def _strict_bp_map_residual(
    ket: dict[int, torch.Tensor],
    bra_conj: dict[int, torch.Tensor],
    messages: dict[tuple[int, str], torch.Tensor],
    site_plan: tuple,
    plan: dict[str, Any],
    *,
    reuse_opposite_cavities: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate the frozen, un-damped BP map without changing ``messages``."""
    B = int(next(iter(ket.values())).shape[0])
    _emit_stage_profile_event("bp_strict_residual_pass", "begin", batch_rows=B)
    device = next(iter(ket.values())).device
    projective = torch.zeros(B, dtype=torch.float64, device=device)
    raw_map = torch.zeros_like(projective)
    bad = torch.zeros(B, dtype=torch.bool, device=device)
    for site, entries in site_plan:
        raw_by_key = _contract_bp_site_messages(
            ket,
            bra_conj,
            messages,
            site,
            entries,
            plan,
            reuse_opposite_cavities=reuse_opposite_cavities,
        )
        for _, direction, _, _ in entries:
            key = (site, direction)
            current = messages[key]
            raw_fixed, bad_fixed, _ = _normalize_message(raw_by_key[key])
            raw_fixed = _phase_align(raw_fixed, current)
            projective = torch.maximum(
                projective,
                _projective_residual(raw_fixed, current).to(torch.float64),
            )
            raw_map = torch.maximum(
                raw_map,
                (raw_fixed - current)
                .abs()
                .reshape(B, -1)
                .max(dim=1)
                .values.to(torch.float64),
            )
            bad |= bad_fixed
    _emit_stage_profile_event("bp_strict_residual_pass", "end", batch_rows=B)
    return projective, raw_map, bad


def _solve_bp(
    ket: dict[int, torch.Tensor],
    bra: dict[int, torch.Tensor],
    initial: dict[tuple[int, str], torch.Tensor],
    plan: dict[str, Any],
    *,
    max_iter: int,
    tol: float,
    damping: float,
    residual_check_interval: int = 1,
    step_residual_gate_factor: float | None = 100.0,
    reuse_opposite_cavities: bool = True,
    active_compaction_ratio: float = 0.5,
    branch_ids: torch.Tensor | None = None,
    allow_nonconverged: bool = False,
) -> tuple[dict[tuple[int, str], torch.Tensor], dict[str, Any]]:
    """Batched in-place Gauss-Seidel BP with strict gated convergence checks.

    The strict projective residual is always evaluated before a branch is
    declared converged.  Scheduled checks are restricted to rows whose cheap
    sweep step is near tolerance; the final sweep checks every active row.
    Once enough rows have converged, the remaining active rows are compacted
    into a smaller physical/message batch; returned messages retain the
    original branch order.
    """
    if residual_check_interval < 1:
        raise ValueError("residual_check_interval must be positive")
    if step_residual_gate_factor is not None and (
        not math.isfinite(step_residual_gate_factor) or step_residual_gate_factor <= 0.0
    ):
        raise ValueError(
            "step_residual_gate_factor must be positive and finite or None"
        )
    if not 0.0 < active_compaction_ratio <= 1.0:
        raise ValueError("active_compaction_ratio must lie in (0, 1]")
    B = int(next(iter(ket.values())).shape[0])
    device = next(iter(ket.values())).device
    work_ket = ket
    work_messages = initial
    del initial
    work_positions = torch.arange(B, dtype=torch.long, device=device)
    active = torch.ones(B, dtype=torch.bool, device=device)
    output_messages = {
        key: torch.empty_like(value) for key, value in work_messages.items()
    }
    iterations_per_branch = torch.zeros(B, dtype=torch.long, device=device)
    projective_residual_per_branch = torch.full(
        (B,), float("inf"), dtype=torch.float64, device=device
    )
    raw_map_residual_per_branch = torch.full_like(
        projective_residual_per_branch, float("inf")
    )
    step_residual_per_branch = torch.full_like(
        projective_residual_per_branch, float("inf")
    )
    site_plan = _bp_site_plan(plan)
    bra_conj = {site: value.conj() for site, value in bra.items()}
    strict_residual_checks = 0
    strict_residual_rows = 0
    strict_residual_gate_skips = 0
    active_compactions = 0

    def branch_id_list(global_positions: torch.Tensor) -> list[int]:
        values = (
            branch_ids.index_select(0, global_positions)
            if branch_ids is not None
            else global_positions
        )
        return [int(value) for value in values.detach().cpu().tolist()]

    def store_rows(local_positions: torch.Tensor) -> torch.Tensor:
        global_positions = work_positions.index_select(0, local_positions)
        for key, value in work_messages.items():
            output_messages[key].index_copy_(
                0, global_positions, value.index_select(0, local_positions)
            )
        return global_positions

    def info(iterations: int) -> dict[str, Any]:
        return {
            "iterations": iterations,
            "iterations_per_branch": (iterations_per_branch.detach().cpu().tolist()),
            "residual_per_branch": (
                projective_residual_per_branch.detach().cpu().tolist()
            ),
            "projective_raw_residual_per_branch": (
                projective_residual_per_branch.detach().cpu().tolist()
            ),
            "raw_map_residual_per_branch": (
                raw_map_residual_per_branch.detach().cpu().tolist()
            ),
            "step_residual_per_branch": (
                step_residual_per_branch.detach().cpu().tolist()
            ),
            "strict_residual_checks": strict_residual_checks,
            "strict_residual_rows": strict_residual_rows,
            "strict_residual_gate_skips": strict_residual_gate_skips,
            "active_compactions": active_compactions,
        }

    for iteration in range(1, max_iter + 1):
        local_B = int(work_positions.numel())
        step_residual = torch.zeros(local_B, dtype=torch.float64, device=device)
        bad_any = torch.zeros(local_B, dtype=torch.bool, device=device)
        _emit_stage_profile_event(
            "bp_ordinary_sweep", "begin", iteration=iteration, batch_rows=local_B
        )
        for site, entries in site_plan:
            raw_by_key = _contract_bp_site_messages(
                work_ket,
                bra_conj,
                work_messages,
                site,
                entries,
                plan,
                reuse_opposite_cavities=reuse_opposite_cavities,
            )
            for _, direction, _, _ in entries:
                key = (site, direction)
                old = work_messages[key]
                raw, bad_raw, _ = _normalize_message(raw_by_key[key])
                raw = _phase_align(raw, old)

                updated = (1.0 - damping) * raw + damping * old
                updated, bad_updated, _ = _normalize_message(updated)
                step_edge_residual = (
                    (updated - old).abs().reshape(local_B, -1).max(dim=1).values
                )
                step_residual = torch.maximum(
                    step_residual,
                    torch.where(
                        active,
                        step_edge_residual.to(torch.float64),
                        torch.zeros_like(step_residual),
                    ),
                )
                work_messages[key] = torch.where(
                    active.reshape(local_B, 1, 1), updated, old
                )
                bad_any |= (bad_raw | bad_updated) & active
        _emit_stage_profile_event(
            "bp_ordinary_sweep", "end", iteration=iteration, batch_rows=local_B
        )

        strict_check = (
            iteration == 1
            or iteration % residual_check_interval == 0
            or iteration == max_iter
        )
        if not strict_check:
            continue
        if bool(bad_any.any().item()):
            bad_local = torch.nonzero(bad_any, as_tuple=False).flatten()
            raise RuntimeError(
                "zero/non-finite single-site BP messages at "
                f"branch_ids={branch_id_list(work_positions.index_select(0, bad_local))}"
            )

        strict_candidate = active.clone()
        if iteration != max_iter and step_residual_gate_factor is not None:
            strict_candidate &= step_residual <= float(step_residual_gate_factor) * tol
        candidate_local = torch.nonzero(strict_candidate, as_tuple=False).flatten()
        if candidate_local.numel() == 0:
            strict_residual_gate_skips += 1
            continue
        strict_residual_checks += 1
        strict_residual_rows += int(candidate_local.numel())

        if int(candidate_local.numel()) == local_B:
            strict_ket = work_ket
            strict_messages = work_messages
        else:
            strict_ket = {
                site: value.index_select(0, candidate_local)
                for site, value in work_ket.items()
            }
            strict_messages = {
                key: value.index_select(0, candidate_local)
                for key, value in work_messages.items()
            }
        strict_B = int(candidate_local.numel())

        # Re-evaluate the un-damped frozen BP map after the complete in-place
        # sweep.  A branch is accepted only against this strict residual.
        strict_projective, strict_raw_map, strict_bad = _strict_bp_map_residual(
            strict_ket,
            bra_conj,
            strict_messages,
            site_plan,
            plan,
            reuse_opposite_cavities=reuse_opposite_cavities,
        )

        if bool(strict_bad.any().item()):
            bad_candidate = torch.nonzero(strict_bad, as_tuple=False).flatten()
            bad_local = candidate_local.index_select(0, bad_candidate)
            raise RuntimeError(
                "zero/non-finite single-site BP messages at "
                f"branch_ids={branch_id_list(work_positions.index_select(0, bad_local))}"
            )

        projective_residual = torch.full(
            (local_B,), float("inf"), dtype=torch.float64, device=device
        )
        raw_map_residual = torch.full_like(projective_residual, float("inf"))
        projective_residual.index_copy_(0, candidate_local, strict_projective)
        raw_map_residual.index_copy_(0, candidate_local, strict_raw_map)
        converged = torch.zeros(local_B, dtype=torch.bool, device=device)
        converged.index_copy_(0, candidate_local, strict_projective < tol)
        converged &= active & ~bad_any
        converged_local = torch.nonzero(converged, as_tuple=False).flatten()
        if converged_local.numel():
            converged_global = store_rows(converged_local)
            iterations_per_branch.index_fill_(0, converged_global, iteration)
            projective_residual_per_branch.index_copy_(
                0,
                converged_global,
                projective_residual.index_select(0, converged_local),
            )
            raw_map_residual_per_branch.index_copy_(
                0, converged_global, raw_map_residual.index_select(0, converged_local)
            )
            step_residual_per_branch.index_copy_(
                0, converged_global, step_residual.index_select(0, converged_local)
            )
        active &= ~converged
        active_count = int(active.sum().item())
        if active_count == 0:
            return output_messages, info(iteration)

        threshold = max(1, int(math.floor(local_B * active_compaction_ratio)))
        if (
            iteration < max_iter
            and active_count < local_B
            and active_count <= threshold
        ):
            _emit_stage_profile_event(
                "bp_active_compaction",
                "begin",
                iteration=iteration,
                rows_before=local_B,
                rows_after=active_count,
            )
            keep = torch.nonzero(active, as_tuple=False).flatten()
            work_ket = {
                site: value.index_select(0, keep) for site, value in work_ket.items()
            }
            work_messages = {
                key: value.index_select(0, keep) for key, value in work_messages.items()
            }
            work_positions = work_positions.index_select(0, keep)
            active = torch.ones(active_count, dtype=torch.bool, device=device)
            active_compactions += 1
            _emit_stage_profile_event(
                "bp_active_compaction",
                "end",
                iteration=iteration,
                rows_before=local_B,
                rows_after=active_count,
            )

    failed_local = torch.nonzero(active, as_tuple=False).flatten()
    failed_global = store_rows(failed_local)
    iterations_per_branch.index_fill_(0, failed_global, max_iter)
    projective_residual_per_branch.index_copy_(
        0, failed_global, projective_residual.index_select(0, failed_local)
    )
    raw_map_residual_per_branch.index_copy_(
        0, failed_global, raw_map_residual.index_select(0, failed_local)
    )
    step_residual_per_branch.index_copy_(
        0, failed_global, step_residual.index_select(0, failed_local)
    )
    ids = branch_id_list(failed_global)
    if allow_nonconverged:
        result_info = info(max_iter)
        result_info.update(
            {
                "failed_positions": failed_global.detach().cpu().tolist(),
                "failed_branch_ids": ids,
            }
        )
        return output_messages, result_info
    raise RuntimeError(
        f"single-site BP failed after {max_iter} sweeps; "
        f"branch_ids={ids}, "
        f"max_residual={projective_residual[failed_local].max().item():.3e} "
        f"(projective_raw), "
        f"max_raw_map_residual="
        f"{raw_map_residual[failed_local].max().item():.3e}, "
        f"max_step_residual={step_residual[failed_local].max().item():.3e}"
    )


def _batched_bethe_site_factors(
    ket: dict[int, torch.Tensor],
    bra: dict[int, torch.Tensor],
    messages: dict[tuple[int, str], torch.Tensor],
    plan: dict[str, Any],
) -> dict[int, torch.Tensor]:
    """Contract same-shape single-site beliefs over a site batch axis."""
    B = int(next(iter(ket.values())).shape[0])
    groups: dict[tuple[Any, ...], list[int]] = {}
    for site in sorted(ket):
        directions = tuple(
            direction for direction in "UDLR" if direction in plan["neighbors"][site]
        )
        bra_shape = (
            tuple(int(x) for x in bra[site].shape[1:])
            if bra[site].ndim == ket[site].ndim
            else tuple(int(x) for x in bra[site].shape)
        )
        signature = (
            directions,
            tuple(int(x) for x in ket[site].shape[1:]),
            bra_shape,
            tuple(
                tuple(
                    int(x)
                    for x in messages[
                        (
                            plan["neighbors"][site][direction],
                            OPPOSITE[direction],
                        )
                    ].shape[1:]
                )
                for direction in directions
            ),
        )
        groups.setdefault(signature, []).append(site)

    site_factors: dict[int, torch.Tensor] = {}
    site_step = int(plan["bethe_site_batch_size"])
    for (directions, _, _, _), sites in groups.items():
        for start in range(0, len(sites), site_step):
            selected = sites[start : start + site_step]
            if len(selected) == 1:
                site = selected[0]
                physical_belief = _contract_site(
                    ket, bra, site, messages, plan, keep_phys=True
                )
                site_factors[site] = torch.diagonal(
                    physical_belief, dim1=1, dim2=2
                ).sum(dim=1)
                continue

            synthetic_ket = {
                0: torch.cat([ket[site] for site in selected], dim=0),
            }
            bra_parts = []
            for site in selected:
                value = bra[site]
                if value.ndim == ket[site].ndim:
                    if int(value.shape[0]) != B:
                        raise ValueError("batched Bethe bra has wrong batch size")
                    bra_parts.append(value)
                else:
                    bra_parts.append(value.unsqueeze(0).expand(B, *value.shape))
            synthetic_bra = {0: torch.cat(bra_parts, dim=0)}
            synthetic_neighbors = {0: {}}
            synthetic_messages = {}
            for ghost, direction in enumerate(directions, start=1):
                synthetic_neighbors[0][direction] = ghost
                incoming_parts = []
                for site in selected:
                    other = plan["neighbors"][site][direction]
                    incoming_parts.append(messages[(other, OPPOSITE[direction])])
                synthetic_messages[(ghost, OPPOSITE[direction])] = torch.cat(
                    incoming_parts, dim=0
                )
            synthetic_plan = {
                "neighbors": synthetic_neighbors,
                "memory_limit": plan["memory_limit"],
            }
            physical_belief = _contract_site(
                synthetic_ket,
                synthetic_bra,
                0,
                synthetic_messages,
                synthetic_plan,
                keep_phys=True,
            )
            factors = torch.diagonal(physical_belief, dim1=1, dim2=2).sum(dim=1)
            factors = factors.reshape(len(selected), B)
            for local, site in enumerate(selected):
                site_factors[site] = factors[local]
    return {site: site_factors[site] for site in sorted(site_factors)}


def _batched_bethe_edge_factors(
    messages: dict[tuple[int, str], torch.Tensor],
    plan: dict[str, Any],
) -> dict[tuple[int, int], torch.Tensor]:
    """Contract equal-shape separator overlaps over an edge batch axis."""
    groups: dict[tuple[Any, ...], list[tuple[int, int, str]]] = {}
    for first, second in plan["undirected_edges"]:
        direction = next(
            side for side, other in plan["neighbors"][first].items() if other == second
        )
        reverse = OPPOSITE[direction]
        key = (
            tuple(int(x) for x in messages[(first, direction)].shape[1:]),
            tuple(int(x) for x in messages[(second, reverse)].shape[1:]),
        )
        groups.setdefault(key, []).append((first, second, direction))

    B = int(next(iter(messages.values())).shape[0])
    edge_factors: dict[tuple[int, int], torch.Tensor] = {}
    edge_step = int(plan["bethe_site_batch_size"])
    for entries in groups.values():
        for start in range(0, len(entries), edge_step):
            selected = entries[start : start + edge_step]
            left = torch.cat(
                [messages[(first, direction)] for first, _, direction in selected],
                dim=0,
            )
            right = torch.cat(
                [
                    messages[(second, OPPOSITE[direction])]
                    for _, second, direction in selected
                ],
                dim=0,
            )
            values = torch.einsum("zab,zab->z", left, right).reshape(len(selected), B)
            for local, (first, second, _) in enumerate(selected):
                edge_factors[(first, second)] = values[local]
    return {edge: edge_factors[edge] for edge in plan["undirected_edges"]}


def _bethe_factors(
    ket: dict[int, torch.Tensor],
    bra: dict[int, torch.Tensor],
    messages: dict[tuple[int, str], torch.Tensor],
    plan: dict[str, Any],
) -> dict[str, Any]:
    """Return the site/edge factors and single-site Bethe partition."""
    # Use exactly the same contraction and physical trace order as the local
    # identity readout.  The direct scalar einsum is algebraically equivalent,
    # but near-traceless mixed branches can lose several relative digits when
    # the contraction optimizer chooses a different summation order.  A strict
    # region identity must compare one numerical definition of z_i, not two
    # differently rounded realizations of it.
    if plan.get("batch_bethe_sites", False):
        site_factors = _batched_bethe_site_factors(ket, bra, messages, plan)
        edge_factors = _batched_bethe_edge_factors(messages, plan)
    else:
        site_factors = {}
        for site in sorted(ket):
            physical_belief = _contract_site(
                ket, bra, site, messages, plan, keep_phys=True
            )
            site_factors[site] = torch.diagonal(physical_belief, dim1=1, dim2=2).sum(
                dim=1
            )
        edge_factors = {}
        for first, second in plan["undirected_edges"]:
            direction = next(
                side
                for side, other in plan["neighbors"][first].items()
                if other == second
            )
            reverse = OPPOSITE[direction]
            edge_factors[(first, second)] = torch.einsum(
                "zab,zab->z", messages[(first, direction)], messages[(second, reverse)]
            )

    for edge, value in edge_factors.items():
        if bool(((~torch.isfinite(value)) | (value == 0)).any()):
            raise RuntimeError(f"single-site separator {edge}: zero/non-finite")

    ref = next(iter(site_factors.values()))
    partition = torch.ones_like(ref)
    for value in site_factors.values():
        if bool((~torch.isfinite(value)).any()):
            raise RuntimeError("non-finite single-site factor")
        partition *= value
    for value in edge_factors.values():
        partition /= value
    if bool((~torch.isfinite(partition)).any()):
        raise RuntimeError("non-finite single-site Bethe partition")
    return {
        "Z": partition,
        "site": site_factors,
        "edge": edge_factors,
    }


def _slice_bethe_factors(
    factors: dict[str, Any],
    start: int,
    stop: int,
) -> dict[str, Any]:
    """View one branch interval of already-computed Bethe factors."""
    return {
        "Z": factors["Z"][start:stop],
        "site": {site: value[start:stop] for site, value in factors["site"].items()},
        "edge": {edge: value[start:stop] for edge, value in factors["edge"].items()},
    }


def _rectangular_patch(
    anchor: tuple[int, ...],
    rows: int,
    cols: int,
    plan: dict[str, Any],
) -> tuple[int, ...]:
    """Place one fixed rectangle inside the lattice while containing anchor."""
    width, length = plan["width"], plan["length"]
    if rows > length or cols > width:
        raise ValueError(
            f"readout rectangle {rows}x{cols} exceeds {length}x{width} lattice"
        )
    coordinates = tuple(divmod(site, width) for site in anchor)
    anchor_rows, anchor_cols = zip(*coordinates)
    if max(anchor_rows) - min(anchor_rows) >= rows:
        raise ValueError(f"anchor {anchor} does not fit {rows}x{cols} readout")
    if max(anchor_cols) - min(anchor_cols) >= cols:
        raise ValueError(f"anchor {anchor} does not fit {rows}x{cols} readout")
    top = min(min(anchor_rows), length - rows)
    left = min(min(anchor_cols), width - cols)
    patch = tuple(
        row * width + col
        for row in range(top, top + rows)
        for col in range(left, left + cols)
    )
    if not set(anchor).issubset(patch):
        raise RuntimeError(f"anchor {anchor} escaped readout patch {patch}")
    return patch


def _select_readout_patch(
    anchor: tuple[int, ...],
    plan: dict[str, Any],
) -> tuple[int, ...]:
    """Select the target support or its nearest 2x2 cavity."""
    block = plan["readout_block"]
    if block in {"support", "strip"}:
        return tuple(anchor)
    if block != "2x2":
        raise RuntimeError(f"unknown readout block {block}")
    width, length = plan["width"], plan["length"]
    coordinates = tuple(divmod(site, width) for site in anchor)
    horizontal = len(coordinates) >= 2 and coordinates[0][0] == coordinates[1][0]
    vertical = len(coordinates) >= 2 and coordinates[0][1] == coordinates[1][1]
    if len(coordinates) >= 2 and not (horizontal or vertical):
        raise ValueError(f"readout anchor {anchor} is not nearest-neighbour")
    if len(coordinates) < 2:
        horizontal = width >= 2
        vertical = not horizontal

    if length < 2 or width < 2:
        raise ValueError("2x2 readout requires a lattice at least 2x2")
    return _rectangular_patch(anchor, 2, 2, plan)


def _patch_side_sites(
    patch: tuple[int, ...],
    direction: str,
    plan: dict[str, Any],
) -> tuple[int, ...]:
    width = plan["width"]
    coordinates = {site: divmod(site, width) for site in patch}
    rows = sorted({row for row, _ in coordinates.values()})
    cols = sorted({col for _, col in coordinates.values()})
    expected = {row * width + col for row in rows for col in cols}
    if expected != set(patch):
        raise ValueError("readout patch must be rectangular")
    if direction == "U":
        return tuple(rows[0] * width + col for col in cols)
    if direction == "D":
        return tuple(rows[-1] * width + col for col in cols)
    if direction == "L":
        return tuple(row * width + cols[0] for row in rows)
    if direction == "R":
        return tuple(row * width + cols[-1] for row in rows)
    raise ValueError(f"unknown direction {direction}")


def _joint_side_message(
    ket: dict[int, torch.Tensor],
    bra: dict[int, torch.Tensor],
    messages: dict[tuple[int, str], torch.Tensor],
    patch_sites: tuple[int, ...],
    direction: str,
    plan: dict[str, Any],
) -> torch.Tensor | None:
    """Contract an adjacent halo pair into one joint boundary message."""
    if len(patch_sites) != 2:
        raise ValueError("joint boundary messages are restricted to two legs")
    halo = tuple(plan["neighbors"][site].get(direction) for site in patch_sites)
    if halo[0] is None or halo[1] is None:
        return None
    first, second = int(halo[0]), int(halo[1])
    internal = next(
        (side for side, other in plan["neighbors"][first].items() if other == second),
        None,
    )
    if internal is None:
        raise RuntimeError(f"non-adjacent halo sites {halo}")

    # Leave the two patch-facing doubled legs open and contract the halo pair's
    # shared doubled bond exactly.  Every other leg sees the frozen global
    # single-site cavity field.
    ket_labels = [list("udlr"), list("abce")]
    bra_labels = [[label.upper() for label in value] for value in ket_labels]
    reverse_internal = OPPOSITE[internal]
    ket_labels[0][DIR_TO_AXIS[internal]] = "i"
    bra_labels[0][DIR_TO_AXIS[internal]] = "I"
    ket_labels[1][DIR_TO_AXIS[reverse_internal]] = "i"
    bra_labels[1][DIR_TO_AXIS[reverse_internal]] = "I"
    toward_patch = OPPOSITE[direction]

    fragments = []
    operands = []
    for local, site in enumerate((first, second)):
        physical = "p" if local == 0 else "q"
        fragments.append("z " + " ".join((*ket_labels[local], physical)))
        operands.append(ket[site])
        fragments.append(" ".join((*bra_labels[local], physical)))
        operands.append(bra[site].conj())
        partner_direction = internal if local == 0 else reverse_internal
        for side, other in plan["neighbors"][site].items():
            if side in (toward_patch, partner_direction):
                continue
            label = ket_labels[local][DIR_TO_AXIS[side]]
            fragments.append(f"z {label} {label.upper()}")
            operands.append(messages[(other, OPPOSITE[side])])

    output = ["z"]
    for local in range(2):
        label = ket_labels[local][DIR_TO_AXIS[toward_patch]]
        output.extend((label, label.upper()))
    value = _einsum(
        ", ".join(fragments) + " -> " + " ".join(output),
        *operands,
        memory_limit=plan["memory_limit"],
    )
    norm = torch.linalg.norm(value.reshape(value.shape[0], -1), dim=1)
    if bool(((~torch.isfinite(norm)) | (norm <= 1e-30)).any()):
        raise RuntimeError(
            f"zero/non-finite joint boundary message on side {direction}"
        )
    return value


def _joint_boundary_messages(
    ket: dict[int, torch.Tensor],
    bra: dict[int, torch.Tensor],
    messages: dict[tuple[int, str], torch.Tensor],
    patch: tuple[int, ...],
    support: tuple[int, ...],
    plan: dict[str, Any],
) -> dict[str, tuple[tuple[int, int], torch.Tensor]]:
    """Build target-adjacent pair cavities without a dense three-leg tensor."""
    joint = {}
    for direction in "UDLR":
        if len(_patch_side_sites(patch, direction, plan)) < 2:
            continue
        side_sites = _joint_side_patch_sites(patch, support, direction, plan)
        value = _joint_side_message(ket, bra, messages, side_sites, direction, plan)
        if value is not None:
            joint[direction] = (side_sites, value)
    return joint


def _joint_side_patch_sites(
    patch: tuple[int, ...],
    support: tuple[int, ...],
    direction: str,
    plan: dict[str, Any],
) -> tuple[int, int]:
    """Select the adjacent boundary pair closest to the readout anchor."""
    side_sites = _patch_side_sites(patch, direction, plan)
    if len(side_sites) == 2:
        return side_sites
    if len(side_sites) < 2:
        raise RuntimeError(f"side {direction} of patch {patch} has one leg")
    axis = 1 if direction in "UD" else 0
    target_center = sum(divmod(site, plan["width"])[axis] for site in support) / len(
        support
    )
    candidates = tuple(zip(side_sites[:-1], side_sites[1:]))
    return min(
        candidates,
        key=lambda pair: abs(
            sum(divmod(site, plan["width"])[axis] for site in pair) / 2 - target_center
        ),
    )


def _contract_labeled_pairwise(
    operands: list[torch.Tensor],
    labels: list[list[int]],
    output_labels: list[int],
    *,
    batch_label: int,
    memory_limit: int,
    trace_labels: set[int],
    description: str,
) -> torch.Tensor:
    """Contract an exact labeled network with a cached pairwise tree."""
    signature = (
        tuple(tuple(int(x) for x in operand.shape) for operand in operands),
        tuple(tuple(operand_labels) for operand_labels in labels),
        tuple(output_labels),
        int(batch_label),
        int(memory_limit),
        tuple(sorted(trace_labels)),
    )

    def reduce_operand(index: int, result_labels: list[int]) -> None:
        operand_labels = labels[index]
        compact = {label: local for local, label in enumerate(operand_labels)}
        operands[index] = torch.einsum(
            operands[index],
            [compact[label] for label in operand_labels],
            [compact[label] for label in result_labels],
        )
        labels[index] = list(result_labels)

    def contract_pair(
        left: int,
        right: int,
        result_labels: list[int],
    ) -> None:
        compact = {
            label: local
            for local, label in enumerate(dict.fromkeys(labels[left] + labels[right]))
        }
        left_subscripts = [compact[label] for label in labels[left]]
        right_subscripts = [compact[label] for label in labels[right]]
        output_subscripts = [compact[label] for label in result_labels]
        combined = torch.einsum(
            operands[left],
            left_subscripts,
            operands[right],
            right_subscripts,
            output_subscripts,
        )
        operands[left] = combined
        labels[left] = list(result_labels)
        del operands[right]
        del labels[right]

    cached_actions = _LABELED_PAIRWISE_TREE_CACHE.get(signature)
    if cached_actions is not None:
        for action in cached_actions:
            if action[0] == "reduce":
                reduce_operand(int(action[1]), list(action[2]))
            elif action[0] == "pair":
                contract_pair(int(action[1]), int(action[2]), list(action[3]))
            else:
                raise RuntimeError(f"unknown cached contraction action {action}")
    else:
        actions: list[tuple[Any, ...]] = []
        output_label_set = set(output_labels)
        while len(operands) > 1:
            occurrences: dict[int, int] = {}
            for operand_labels in labels:
                for label in operand_labels:
                    occurrences[label] = occurrences.get(label, 0) + 1

            reduced_singleton = False
            for index, operand_labels in enumerate(labels):
                singleton = [
                    label
                    for label in operand_labels
                    if occurrences[label] == 1 and label not in output_label_set
                ]
                if not singleton:
                    continue
                result_labels = [
                    label for label in operand_labels if label not in singleton
                ]
                reduce_operand(index, result_labels)
                actions.append(("reduce", index, tuple(result_labels)))
                reduced_singleton = True
                break
            if reduced_singleton:
                continue

            candidates = []
            for left in range(len(operands)):
                left_labels = set(labels[left])
                for right in range(left + 1, len(operands)):
                    shared_nonbatch = (left_labels & set(labels[right])) - {batch_label}
                    if not shared_nonbatch:
                        continue
                    contract_labels = shared_nonbatch - output_label_set
                    if any(occurrences[label] > 2 for label in contract_labels):
                        continue
                    result_labels = []
                    for label in labels[left] + labels[right]:
                        if label in shared_nonbatch and label not in output_label_set:
                            continue
                        if label not in result_labels:
                            result_labels.append(label)
                    dimensions: dict[int, int] = {}
                    for operand_labels, operand in (
                        (labels[left], operands[left]),
                        (labels[right], operands[right]),
                    ):
                        for label, dimension in zip(operand_labels, operand.shape):
                            old = dimensions.setdefault(label, int(dimension))
                            if old != int(dimension):
                                raise RuntimeError(
                                    f"incompatible dimension for label {label}: "
                                    f"{old} != {int(dimension)}"
                                )
                    result_elements = math.prod(
                        dimensions[label] for label in result_labels
                    )
                    admissible = result_elements <= memory_limit
                    trace_only = (
                        bool(shared_nonbatch) and shared_nonbatch <= trace_labels
                    )
                    candidates.append(
                        (
                            10**12 if trace_only else 1,
                            result_elements,
                            len(result_labels),
                            left,
                            right,
                            result_labels,
                            admissible,
                        )
                    )
            if not candidates:
                raise RuntimeError(
                    f"{description} lost a pairwise-connected network path"
                )
            candidates.sort(
                key=lambda item: (item[0] * item[1], item[1], item[2], item[3], item[4])
            )
            selected = next((item for item in candidates if item[6]), None)
            if selected is None:
                smallest = min(candidates, key=lambda item: item[1])
                diagnostic = ""
                if description.startswith("matrix-free 2x2"):
                    diagnostic = (
                        "; state="
                        + repr(
                            [
                                (
                                    tuple(int(x) for x in operand.shape),
                                    tuple(operand_labels),
                                )
                                for operand, operand_labels in zip(operands, labels)
                            ]
                        )
                        + "; actions="
                        + repr(actions)
                        + "; smallest_pair="
                        + repr(
                            (
                                int(smallest[3]),
                                int(smallest[4]),
                                tuple(smallest[5]),
                            )
                        )
                    )
                raise RuntimeError(
                    f"{description} exceeds einsum memory limit: "
                    f"{smallest[1]} elements > {memory_limit}"
                    f"{diagnostic}"
                )
            _, _, _, left, right, result_labels, _ = selected
            actions.append(("pair", left, right, tuple(result_labels)))
            contract_pair(left, right, result_labels)

        if len(_LABELED_PAIRWISE_TREE_CACHE) >= 4096:
            _LABELED_PAIRWISE_TREE_CACHE.clear()
        _LABELED_PAIRWISE_TREE_CACHE[signature] = tuple(actions)

    output_label_set = set(output_labels)
    final = operands[0]
    final_labels = labels[0]
    singleton = [label for label in final_labels if label not in output_label_set]
    if singleton:
        result_labels = [label for label in final_labels if label in output_label_set]
        compact = {label: local for local, label in enumerate(final_labels)}
        final = torch.einsum(
            final,
            [compact[label] for label in final_labels],
            [compact[label] for label in result_labels],
        )
        final_labels = result_labels
    if set(final_labels) != output_label_set:
        raise RuntimeError(
            f"{description} did not close all labels: "
            f"final={final_labels}, output={output_labels}"
        )
    if final_labels != output_labels:
        final = final.permute([final_labels.index(label) for label in output_labels])
    return final


def _labeled_greedy_path_exists(
    operand_shapes: list[tuple[int, ...]],
    labels: list[list[int]],
    output_labels: list[int],
    *,
    batch_label: int,
    memory_limit: int,
    trace_labels: set[int],
) -> bool:
    """Preflight the historical greedy tree without allocating tensors."""
    state_labels = [list(value) for value in labels]
    dimensions: dict[int, int] = {}
    for shape, operand_labels in zip(operand_shapes, state_labels):
        for label, dimension in zip(operand_labels, shape):
            old = dimensions.setdefault(label, int(dimension))
            if old != int(dimension):
                raise RuntimeError(
                    f"incompatible dimension for label {label}: "
                    f"{old} != {int(dimension)}"
                )
    output_label_set = set(output_labels)
    while len(state_labels) > 1:
        occurrences: dict[int, int] = {}
        for operand_labels in state_labels:
            for label in operand_labels:
                occurrences[label] = occurrences.get(label, 0) + 1

        reduced_singleton = False
        for index, operand_labels in enumerate(state_labels):
            singleton = [
                label
                for label in operand_labels
                if occurrences[label] == 1 and label not in output_label_set
            ]
            if singleton:
                state_labels[index] = [
                    label for label in operand_labels if label not in singleton
                ]
                reduced_singleton = True
                break
        if reduced_singleton:
            continue

        candidates = []
        for left in range(len(state_labels)):
            left_labels = set(state_labels[left])
            for right in range(left + 1, len(state_labels)):
                shared_nonbatch = (left_labels & set(state_labels[right])) - {
                    batch_label
                }
                if not shared_nonbatch:
                    continue
                contract_labels = shared_nonbatch - output_label_set
                if any(occurrences[label] > 2 for label in contract_labels):
                    continue
                result_labels = []
                for label in state_labels[left] + state_labels[right]:
                    if label in shared_nonbatch and label not in output_label_set:
                        continue
                    if label not in result_labels:
                        result_labels.append(label)
                result_elements = math.prod(
                    dimensions[label] for label in result_labels
                )
                trace_only = bool(shared_nonbatch) and shared_nonbatch <= trace_labels
                candidates.append(
                    (
                        10**12 if trace_only else 1,
                        result_elements,
                        len(result_labels),
                        left,
                        right,
                        result_labels,
                    )
                )
        if not candidates:
            return False
        candidates.sort(
            key=lambda item: (item[0] * item[1], item[1], item[2], item[3], item[4])
        )
        selected = next((item for item in candidates if item[1] <= memory_limit), None)
        if selected is None:
            return False
        _, _, _, left, right, result_labels = selected
        state_labels[left] = list(result_labels)
        del state_labels[right]
    return True


def _canonicalize_bounded_2x2_network(
    operand_shapes: list[tuple[int, ...]],
    labels: list[list[int]],
    output_labels: list[int],
    *,
    batch_label: int,
    memory_limit: int,
    trace_labels: set[int],
    operand_positions: list[tuple[int, int]],
) -> tuple[tuple[Any, ...], dict[int, int]]:
    """Canonicalize label names and lattice translation for tree reuse.

    Operand order and axis order remain untouched because tree masks refer to
    them directly.  Integer labels are renamed by first occurrence, with the
    batch label fixed at zero.  Absolute lattice coordinates are translated
    to a zero-based bounding box.  These transformations preserve the exact
    tensor network while allowing equivalent patches at different sites to
    share one contraction tree.
    """
    if len(operand_shapes) != len(labels):
        raise ValueError("operand shape/label count mismatch")
    if len(operand_positions) != len(labels):
        raise ValueError("operand position metadata mismatch")
    original_to_canonical = {int(batch_label): 0}
    canonical_to_original = {0: int(batch_label)}
    next_label = 1
    for operand_labels in labels:
        for label in operand_labels:
            label = int(label)
            if label not in original_to_canonical:
                original_to_canonical[label] = next_label
                canonical_to_original[next_label] = label
                next_label += 1
    missing = (
        set(int(value) for value in output_labels)
        | set(int(value) for value in trace_labels)
    ) - set(original_to_canonical)
    if missing:
        raise RuntimeError(
            f"bounded 2x2 metadata references absent labels {sorted(missing)}"
        )
    min_row = min(row for row, _ in operand_positions)
    min_col = min(col for _, col in operand_positions)
    canonical_labels = tuple(
        tuple(original_to_canonical[int(label)] for label in operand_labels)
        for operand_labels in labels
    )
    signature = (
        "bounded_2x2_tree_v1",
        tuple(tuple(int(value) for value in shape) for shape in operand_shapes),
        canonical_labels,
        tuple(original_to_canonical[int(value)] for value in output_labels),
        0,
        int(memory_limit),
        tuple(sorted(original_to_canonical[int(value)] for value in trace_labels)),
        tuple(
            (int(row - min_row), int(col - min_col)) for row, col in operand_positions
        ),
    )
    return signature, canonical_to_original


def _validate_bounded_2x2_tree(
    operand_shapes: tuple[tuple[int, ...], ...],
    labels: tuple[tuple[int, ...], ...],
    output_labels: tuple[int, ...],
    *,
    batch_label: int,
    memory_limit: int,
    actions: tuple[tuple[Any, ...], ...],
) -> int:
    """Validate a serialized tree symbolically before it can be executed."""
    operand_count = len(labels)
    if operand_count < 2 or operand_count > 16:
        raise RuntimeError("invalid bounded tree operand count")
    dimensions: dict[int, int] = {}
    occurrence_masks: dict[int, int] = {}
    for index, (shape, operand_labels) in enumerate(zip(operand_shapes, labels)):
        if len(shape) != len(operand_labels):
            raise RuntimeError("invalid bounded tree operand rank")
        bit = 1 << index
        for label, dimension in zip(operand_labels, shape):
            old = dimensions.setdefault(int(label), int(dimension))
            if old != int(dimension):
                raise RuntimeError("invalid bounded tree label dimension")
            occurrence_masks[int(label)] = occurrence_masks.get(int(label), 0) | bit
    output_set = set(int(value) for value in output_labels)
    full_mask = (1 << operand_count) - 1

    def expected_labels(mask: int) -> tuple[int, ...]:
        return tuple(
            label
            for label in sorted(
                dimensions, key=lambda value: (value != batch_label, value)
            )
            if (occurrence_masks[label] & mask)
            and (
                label in output_set
                or (occurrence_masks[label] & mask) != occurrence_masks[label]
            )
        )

    active = {1 << index for index in range(operand_count)}
    peak = 0
    for action in actions:
        if not action:
            raise RuntimeError("empty bounded tree action")
        if action[0] == "reduce":
            if len(action) != 3:
                raise RuntimeError("invalid bounded reduce action")
            mask = int(action[1])
            result_labels = tuple(int(value) for value in action[2])
            if mask not in active or mask & (mask - 1):
                raise RuntimeError("invalid bounded reduce mask")
            if result_labels != expected_labels(mask):
                raise RuntimeError("invalid bounded reduce labels")
            continue
        if action[0] != "pair" or len(action) != 5:
            raise RuntimeError("invalid bounded pair action")
        left, right, result_mask = map(int, action[1:4])
        result_labels = tuple(int(value) for value in action[4])
        if (
            left not in active
            or right not in active
            or left == right
            or left & right
            or result_mask != (left | right)
        ):
            raise RuntimeError("invalid bounded pair masks")
        shared = set(expected_labels(left)) & set(expected_labels(right))
        if not (shared - {batch_label}):
            raise RuntimeError("disconnected bounded pair action")
        if result_labels != expected_labels(result_mask):
            raise RuntimeError("invalid bounded pair result labels")
        elements = math.prod(dimensions[label] for label in result_labels)
        if elements > memory_limit:
            raise RuntimeError("bounded tree exceeds its memory cap")
        peak = max(peak, int(elements))
        active.remove(left)
        active.remove(right)
        active.add(result_mask)
    if active != {full_mask}:
        raise RuntimeError("bounded tree does not reach a unique root")
    return peak


def _plan_greedy_bounded_2x2_tree(
    operand_shapes: tuple[tuple[int, ...], ...],
    labels: tuple[tuple[int, ...], ...],
    output_labels: tuple[int, ...],
    *,
    batch_label: int,
    memory_limit: int,
    trace_labels: set[int],
) -> tuple[tuple[tuple[Any, ...], ...], int] | None:
    """Build the historical greedy choice as a reusable subset-mask tree."""
    operand_count = len(labels)
    dimensions: dict[int, int] = {}
    occurrence_masks: dict[int, int] = {}
    for index, (shape, operand_labels) in enumerate(zip(operand_shapes, labels)):
        bit = 1 << index
        for label, dimension in zip(operand_labels, shape):
            old = dimensions.setdefault(int(label), int(dimension))
            if old != int(dimension):
                raise RuntimeError("incompatible canonical label dimension")
            occurrence_masks[int(label)] = occurrence_masks.get(int(label), 0) | bit
    output_set = set(int(value) for value in output_labels)
    label_order = sorted(dimensions, key=lambda value: (value != batch_label, value))

    def result_labels(mask: int) -> tuple[int, ...]:
        return tuple(
            label
            for label in label_order
            if (occurrence_masks[label] & mask)
            and (
                label in output_set
                or (occurrence_masks[label] & mask) != occurrence_masks[label]
            )
        )

    states = [1 << index for index in range(operand_count)]
    actions: list[tuple[Any, ...]] = []
    for index, operand_labels in enumerate(labels):
        kept = result_labels(1 << index)
        if tuple(operand_labels) != kept:
            actions.append(("reduce", 1 << index, kept))
    peak = 0
    while len(states) > 1:
        candidates = []
        for left_index, left_mask in enumerate(states):
            left_labels = set(result_labels(left_mask))
            for right_index in range(left_index + 1, len(states)):
                right_mask = states[right_index]
                shared = (left_labels & set(result_labels(right_mask))) - {batch_label}
                if not shared:
                    continue
                merged = left_mask | right_mask
                merged_labels = result_labels(merged)
                elements = math.prod(dimensions[label] for label in merged_labels)
                trace_only = bool(shared) and shared <= trace_labels
                candidates.append(
                    (
                        10**12 if trace_only else 1,
                        elements,
                        len(merged_labels),
                        left_index,
                        right_index,
                        left_mask,
                        right_mask,
                        merged,
                        merged_labels,
                    )
                )
        if not candidates:
            raise RuntimeError("canonical greedy tree lost connectivity")
        candidates.sort(
            key=lambda item: (item[0] * item[1], item[1], item[2], item[3], item[4])
        )
        selected = next((item for item in candidates if item[1] <= memory_limit), None)
        if selected is None:
            return None
        (
            _,
            elements,
            _,
            left_index,
            right_index,
            left_mask,
            right_mask,
            merged,
            merged_labels,
        ) = selected
        actions.append(("pair", left_mask, right_mask, merged, merged_labels))
        states[left_index] = merged
        del states[right_index]
        peak = max(peak, int(elements))
    return tuple(actions), peak


def _score_bounded_2x2_tree(
    operand_shapes: tuple[tuple[int, ...], ...],
    labels: tuple[tuple[int, ...], ...],
    operand_positions: tuple[tuple[int, int], ...],
    actions: tuple[tuple[Any, ...], ...],
) -> tuple[float, float, int, int]:
    """Score actual binary contraction work, peak, and spatial locality."""
    dimensions: dict[int, int] = {}
    occurrence_masks: dict[int, int] = {}
    for index, (shape, operand_labels) in enumerate(zip(operand_shapes, labels)):
        bit = 1 << index
        for label, dimension in zip(operand_labels, shape):
            dimensions[int(label)] = int(dimension)
            occurrence_masks[int(label)] = occurrence_masks.get(int(label), 0) | bit
    active_labels = {
        1 << index: set(int(value) for value in operand_labels)
        for index, operand_labels in enumerate(labels)
    }
    boxes = {
        1 << index: (int(row), int(row), int(col), int(col))
        for index, (row, col) in enumerate(operand_positions)
    }
    max_work_log = 0.0
    total_work_log = 0.0
    peak = 0
    geometry = 0
    for action in actions:
        if action[0] == "reduce":
            mask = int(action[1])
            active_labels[mask] = set(int(value) for value in action[2])
            continue
        left, right, merged = map(int, action[1:4])
        work_log = sum(
            math.log2(dimensions[label])
            for label in active_labels[left] | active_labels[right]
        )
        max_work_log = max(max_work_log, work_log)
        total_work_log += work_log
        merged_labels = set(int(value) for value in action[4])
        elements = math.prod(dimensions[label] for label in merged_labels)
        peak = max(peak, int(elements))
        left_box = boxes[left]
        right_box = boxes[right]
        for box in (left_box, right_box):
            geometry += (box[1] - box[0] + 1) * (box[3] - box[2] + 1)
        boxes[merged] = (
            min(left_box[0], right_box[0]),
            max(left_box[1], right_box[1]),
            min(left_box[2], right_box[2]),
            max(left_box[3], right_box[3]),
        )
        active_labels[merged] = merged_labels
        del active_labels[left]
        del active_labels[right]
        del boxes[left]
        del boxes[right]
    return max_work_log, total_work_log, peak, geometry


def _bounded_2x2_signature_json(signature: tuple[Any, ...]) -> str:
    return json.dumps(signature, separators=(",", ":"), ensure_ascii=True)


def _bounded_2x2_persistent_path() -> Path | None:
    override = os.environ.get("TANGENT_BP_BOUNDED_2X2_TREE_CACHE")
    if override is None:
        # Default to a sidecar beside this engine, independent of whichever
        # benchmark or application imports it.  An explicit empty override
        # disables disk persistence; a non-empty override selects another
        # location without changing the numerical configuration.
        return (
            Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
            / "residual_tn"
            / "bounded_2x2_tree_cache_v1.json"
        )
    value = override.strip()
    return None if not value else Path(value).expanduser()


def _load_bounded_2x2_persistent_cache(
    audit: dict[str, Any] | None = None,
) -> None:
    global _LABELED_2X2_PERSISTENT_CACHE_PATH
    global _LABELED_2X2_PERSISTENT_CACHE_ENTRIES
    path = _bounded_2x2_persistent_path()
    path_key = None if path is None else str(path.resolve())
    if path_key == _LABELED_2X2_PERSISTENT_CACHE_PATH:
        return
    _LABELED_2X2_PERSISTENT_CACHE_PATH = path_key
    _LABELED_2X2_PERSISTENT_CACHE_ENTRIES = dict(_LABELED_2X2_BUILTIN_CACHE_ENTRIES)
    if path is None or not path.exists():
        if audit is not None:
            audit["persistent_cache_entries_loaded"] = len(
                _LABELED_2X2_PERSISTENT_CACHE_ENTRIES
            )
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or payload.get("version") != _LABELED_2X2_PERSISTENT_CACHE_VERSION
            or not isinstance(payload.get("entries"), dict)
        ):
            raise ValueError("unsupported bounded 2x2 cache schema")
        _LABELED_2X2_PERSISTENT_CACHE_ENTRIES.update(payload["entries"])
        if audit is not None:
            audit["persistent_cache_entries_loaded"] = len(
                _LABELED_2X2_PERSISTENT_CACHE_ENTRIES
            )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        if audit is not None:
            audit["persistent_cache_load_errors"] += 1
            audit["persistent_cache_entries_loaded"] = len(
                _LABELED_2X2_PERSISTENT_CACHE_ENTRIES
            )


def flush_bounded_2x2_tree_cache() -> None:
    """Atomically persist newly verified bounded 2x2 contraction trees."""
    global _LABELED_2X2_PERSISTENT_CACHE_DIRTY
    if _LABELED_2X2_PERSISTENT_CACHE_DIRTY == 0:
        return
    path = _bounded_2x2_persistent_path()
    if path is None:
        _LABELED_2X2_PERSISTENT_CACHE_DIRTY = 0
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        entries = dict(_LABELED_2X2_PERSISTENT_CACHE_ENTRIES)
        if path.exists():
            try:
                current = json.loads(path.read_text(encoding="utf-8"))
                if (
                    isinstance(current, dict)
                    and current.get("version") == _LABELED_2X2_PERSISTENT_CACHE_VERSION
                    and isinstance(current.get("entries"), dict)
                ):
                    entries = {**current["entries"], **entries}
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        payload = {
            "version": _LABELED_2X2_PERSISTENT_CACHE_VERSION,
            "entries": entries,
        }
        temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, path)
        _LABELED_2X2_PERSISTENT_CACHE_DIRTY = 0
    except OSError:
        # The cache is a scheduling optimization only.  A read-only or
        # ephemeral job directory must not change the numerical computation.
        return


def _decode_bounded_2x2_actions(value: Any) -> tuple[tuple[Any, ...], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("serialized bounded actions must be a list")
    actions = []
    for item in value:
        if not isinstance(item, (list, tuple)) or not item:
            raise ValueError("invalid serialized bounded action")
        if item[0] == "reduce" and len(item) == 3:
            actions.append(("reduce", int(item[1]), tuple(int(x) for x in item[2])))
        elif item[0] == "pair" and len(item) == 5:
            actions.append(
                (
                    "pair",
                    int(item[1]),
                    int(item[2]),
                    int(item[3]),
                    tuple(int(x) for x in item[4]),
                )
            )
        else:
            raise ValueError("unknown serialized bounded action")
    return tuple(actions)


def _restore_bounded_2x2_action_labels(
    actions: tuple[tuple[Any, ...], ...],
    canonical_to_original: dict[int, int],
) -> tuple[tuple[Any, ...], ...]:
    restored = []
    for action in actions:
        if action[0] == "reduce":
            restored.append(
                (
                    "reduce",
                    int(action[1]),
                    tuple(canonical_to_original[int(value)] for value in action[2]),
                )
            )
        else:
            restored.append(
                (
                    "pair",
                    int(action[1]),
                    int(action[2]),
                    int(action[3]),
                    tuple(canonical_to_original[int(value)] for value in action[4]),
                )
            )
    return tuple(restored)


def _plan_bounded_2x2_tree(
    operand_shapes: list[tuple[int, ...]],
    labels: list[list[int]],
    output_labels: list[int],
    *,
    batch_label: int,
    memory_limit: int,
    operand_positions: list[tuple[int, int]],
    description: str,
) -> tuple[tuple[tuple[Any, ...], ...], int]:
    """Find an exact 2x2 contraction tree under a hard element cap.

    The search is an exact subset dynamic program over at most sixteen local
    factors.  It is invoked only after the cheap historical greedy preflight
    proves that its tree reaches a dead end.  Tensor-network connectivity is
    mandatory at every merge, and lattice bounding boxes break otherwise equal
    choices, so the selected tree remains local and deterministic.  No tensor
    values are inspected and no index is sliced or approximated.
    """
    operand_count = len(labels)
    if operand_count < 2 or operand_count > 16:
        raise RuntimeError(
            f"{description} bounded planner supports 2..16 operands, "
            f"got {operand_count}"
        )
    if len(operand_shapes) != operand_count:
        raise ValueError("operand shape/label count mismatch")
    if len(operand_positions) != operand_count:
        raise ValueError("operand position metadata mismatch")

    dimensions: dict[int, int] = {}
    occurrence_masks: dict[int, int] = {}
    for index, (shape, operand_labels) in enumerate(zip(operand_shapes, labels)):
        if len(shape) != len(operand_labels):
            raise ValueError("operand shape/label rank mismatch")
        if len(set(operand_labels)) != len(operand_labels):
            raise RuntimeError(
                f"{description} bounded planner does not accept repeated "
                "labels within one operand"
            )
        bit = 1 << index
        for label, dimension in zip(operand_labels, shape):
            old = dimensions.setdefault(label, int(dimension))
            if old != int(dimension):
                raise RuntimeError(
                    f"incompatible dimension for label {label}: "
                    f"{old} != {int(dimension)}"
                )
            occurrence_masks[label] = occurrence_masks.get(label, 0) | bit

    output_label_set = set(output_labels)
    full_mask = (1 << operand_count) - 1
    label_order = sorted(dimensions, key=lambda label: (label != batch_label, label))
    result_labels: list[tuple[int, ...]] = [tuple()] * (full_mask + 1)
    result_label_sets: list[frozenset[int]] = [
        frozenset() for _ in range(full_mask + 1)
    ]
    result_elements = [1] * (full_mask + 1)

    min_row = [0] * (full_mask + 1)
    max_row = [0] * (full_mask + 1)
    min_col = [0] * (full_mask + 1)
    max_col = [0] * (full_mask + 1)
    bounding_area = [0] * (full_mask + 1)

    for mask in range(1, full_mask + 1):
        kept = tuple(
            label
            for label in label_order
            if (occurrence_masks[label] & mask)
            and (
                label in output_label_set
                or (occurrence_masks[label] & mask) != occurrence_masks[label]
            )
        )
        result_labels[mask] = kept
        result_label_sets[mask] = frozenset(kept)
        result_elements[mask] = math.prod(dimensions[label] for label in kept)

        least = mask & -mask
        index = least.bit_length() - 1
        row, col = operand_positions[index]
        remainder = mask ^ least
        if remainder == 0:
            min_row[mask] = max_row[mask] = int(row)
            min_col[mask] = max_col[mask] = int(col)
        else:
            min_row[mask] = min(int(row), min_row[remainder])
            max_row[mask] = max(int(row), max_row[remainder])
            min_col[mask] = min(int(col), min_col[remainder])
            max_col[mask] = max(int(col), max_col[remainder])
        bounding_area[mask] = (max_row[mask] - min_row[mask] + 1) * (
            max_col[mask] - min_col[mask] + 1
        )

    valid = [False] * (full_mask + 1)
    peak = [0] * (full_mask + 1)
    max_work_log = [0.0] * (full_mask + 1)
    total_work_log = [0.0] * (full_mask + 1)
    geometry_cost = [0] * (full_mask + 1)
    split: list[tuple[int, int] | None] = [None] * (full_mask + 1)
    for index in range(operand_count):
        valid[1 << index] = True

    for mask in range(1, full_mask + 1):
        if mask & (mask - 1) == 0:
            continue
        if result_elements[mask] > memory_limit:
            continue
        anchor = mask & -mask
        left = (mask - 1) & mask
        best: tuple[Any, ...] | None = None
        while left:
            if left & anchor:
                right = mask ^ left
                if right and valid[left] and valid[right]:
                    shared = (result_label_sets[left] & result_label_sets[right]) - {
                        batch_label
                    }
                    if shared:
                        work_log = sum(
                            math.log2(dimensions[label])
                            for label in (
                                result_label_sets[left] | result_label_sets[right]
                            )
                        )
                        candidate = (
                            max(max_work_log[left], max_work_log[right], work_log),
                            total_work_log[left] + total_work_log[right] + work_log,
                            max(peak[left], peak[right], result_elements[mask]),
                            geometry_cost[left]
                            + geometry_cost[right]
                            + bounding_area[left]
                            + bounding_area[right],
                            max(left.bit_count(), right.bit_count()),
                            left,
                            right,
                        )
                        if best is None or candidate < best:
                            best = candidate
            left = (left - 1) & mask
        if best is None:
            continue
        valid[mask] = True
        max_work_log[mask] = float(best[0])
        total_work_log[mask] = float(best[1])
        peak[mask] = int(best[2])
        geometry_cost[mask] = int(best[3])
        split[mask] = (int(best[5]), int(best[6]))

    if not valid[full_mask]:
        raise RuntimeError(
            f"{description} has no exact connected binary tree with every "
            f"intermediate <= {memory_limit} elements"
        )

    actions: list[tuple[Any, ...]] = []
    for index, operand_labels in enumerate(labels):
        mask = 1 << index
        if tuple(operand_labels) != result_labels[mask]:
            actions.append(("reduce", mask, result_labels[mask]))

    def append_tree(mask: int) -> None:
        children = split[mask]
        if children is None:
            return
        left, right = children
        append_tree(left)
        append_tree(right)
        actions.append(("pair", left, right, mask, result_labels[mask]))

    append_tree(full_mask)
    return tuple(actions), int(peak[full_mask])


def _execute_bounded_2x2_tree(
    operands: list[torch.Tensor],
    labels: list[list[int]],
    output_labels: list[int],
    actions: tuple[tuple[Any, ...], ...],
    *,
    description: str,
) -> torch.Tensor:
    """Execute a preplanned exact subset tree."""
    tensors = {1 << index: operand for index, operand in enumerate(operands)}
    tensor_labels = {
        1 << index: list(operand_labels) for index, operand_labels in enumerate(labels)
    }
    for action in actions:
        if action[0] == "reduce":
            mask = int(action[1])
            result_labels = list(action[2])
            current_labels = tensor_labels[mask]
            compact = {label: local for local, label in enumerate(current_labels)}
            tensors[mask] = torch.einsum(
                tensors[mask],
                [compact[label] for label in current_labels],
                [compact[label] for label in result_labels],
            )
            tensor_labels[mask] = result_labels
            continue
        if action[0] != "pair":
            raise RuntimeError(f"unknown bounded contraction action {action}")
        left, right, result_mask = map(int, action[1:4])
        result_labels = list(action[4])
        left_labels = tensor_labels[left]
        right_labels = tensor_labels[right]
        compact = {
            label: local
            for local, label in enumerate(dict.fromkeys(left_labels + right_labels))
        }
        tensors[result_mask] = torch.einsum(
            tensors[left],
            [compact[label] for label in left_labels],
            tensors[right],
            [compact[label] for label in right_labels],
            [compact[label] for label in result_labels],
        )
        tensor_labels[result_mask] = result_labels
        del tensors[left]
        del tensors[right]
        del tensor_labels[left]
        del tensor_labels[right]

    full_mask = (1 << len(operands)) - 1
    if full_mask not in tensors:
        raise RuntimeError(f"{description} bounded tree did not reach the root")
    final = tensors[full_mask]
    final_labels = tensor_labels[full_mask]
    if set(final_labels) != set(output_labels):
        raise RuntimeError(
            f"{description} bounded tree did not close all labels: "
            f"final={final_labels}, output={output_labels}"
        )
    if final_labels != output_labels:
        final = final.permute([final_labels.index(label) for label in output_labels])
    return final


def _contract_labeled_2x2_bounded(
    operands: list[torch.Tensor],
    labels: list[list[int]],
    output_labels: list[int],
    *,
    batch_label: int,
    memory_limit: int,
    trace_labels: set[int],
    operand_positions: list[tuple[int, int]],
    description: str,
    audit: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, int, bool]:
    """Execute the best cached/planned exact tree under the hard cap.

    The boolean return records whether the historical greedy planner had no
    admissible tree.  Every low-rank 2x2 network uses the canonical persistent
    planner.  By default, a cache miss persists a feasible greedy tree without
    paying exponential online planning cost.  If greedy cannot satisfy the
    cap, DP is still required.  Proactive greedy-versus-DP comparison is an
    explicit opt-in for repeated-run tuning.
    """
    global _LABELED_2X2_PERSISTENT_CACHE_DIRTY
    operand_shapes = [
        tuple(int(value) for value in operand.shape) for operand in operands
    ]
    signature, canonical_to_original = _canonicalize_bounded_2x2_network(
        operand_shapes,
        labels,
        output_labels,
        batch_label=batch_label,
        memory_limit=memory_limit,
        trace_labels=trace_labels,
        operand_positions=operand_positions,
    )
    canonical_shapes = signature[1]
    canonical_labels = signature[2]
    canonical_output = signature[3]
    canonical_batch = int(signature[4])
    canonical_memory_limit = int(signature[5])
    signature_json = _bounded_2x2_signature_json(signature)
    digest = hashlib.sha256(signature_json.encode("ascii")).hexdigest()
    _load_bounded_2x2_persistent_cache(audit)

    cached = _LABELED_2X2_BOUNDED_TREE_CACHE.get(signature)
    newly_planned = False
    if cached is None:
        persistent = _LABELED_2X2_PERSISTENT_CACHE_ENTRIES.get(digest)
        if (
            isinstance(persistent, dict)
            and persistent.get("signature") == signature_json
        ):
            try:
                actions = _decode_bounded_2x2_actions(persistent.get("actions"))
                planned_peak = _validate_bounded_2x2_tree(
                    canonical_shapes,
                    canonical_labels,
                    canonical_output,
                    batch_label=canonical_batch,
                    memory_limit=canonical_memory_limit,
                    actions=actions,
                )
                if int(persistent.get("planned_peak")) != planned_peak:
                    raise ValueError("serialized bounded peak mismatch")
                greedy_feasible = bool(persistent.get("greedy_feasible", False))
                if audit is not None:
                    if digest in _LABELED_2X2_BUILTIN_CACHE_DIGESTS:
                        audit["canonical_builtin_cache_hits"] += 1
                    else:
                        audit["canonical_disk_cache_hits"] += 1
            except (KeyError, TypeError, ValueError, RuntimeError):
                _LABELED_2X2_PERSISTENT_CACHE_ENTRIES.pop(digest, None)
                persistent = None
                if audit is not None:
                    audit["persistent_cache_validation_errors"] += 1
        else:
            persistent = None
        if persistent is None:
            planning_started = time.perf_counter()
            greedy = _plan_greedy_bounded_2x2_tree(
                canonical_shapes,
                canonical_labels,
                canonical_output,
                batch_label=canonical_batch,
                memory_limit=canonical_memory_limit,
                trace_labels=set(int(value) for value in signature[6]),
            )
            greedy_feasible = greedy is not None
            optimize_max_operands = _bounded_2x2_optimize_max_operands()
            run_exact_dp = (
                greedy is None or len(canonical_labels) <= optimize_max_operands
            )
            if not run_exact_dp:
                actions, planned_peak = greedy
                planned_peak = int(planned_peak)
                winner = "greedy"
                if audit is not None:
                    audit["canonical_greedy_only_plans"] += 1
            else:
                dp_actions, dp_peak = _plan_bounded_2x2_tree(
                    list(canonical_shapes),
                    [list(value) for value in canonical_labels],
                    list(canonical_output),
                    batch_label=canonical_batch,
                    memory_limit=canonical_memory_limit,
                    operand_positions=list(signature[7]),
                    description=description,
                )
                dp_score = _score_bounded_2x2_tree(
                    canonical_shapes, canonical_labels, tuple(signature[7]), dp_actions
                )
                if greedy is None:
                    actions, planned_peak = dp_actions, int(dp_peak)
                    winner = "dp"
                else:
                    greedy_actions, greedy_peak = greedy
                    greedy_score = _score_bounded_2x2_tree(
                        canonical_shapes,
                        canonical_labels,
                        tuple(signature[7]),
                        greedy_actions,
                    )
                    if greedy_score < dp_score:
                        actions, planned_peak = (greedy_actions, int(greedy_peak))
                        winner = "greedy"
                    else:
                        actions, planned_peak = dp_actions, int(dp_peak)
                        winner = "dp"
            newly_planned = True
            if audit is not None:
                audit["canonical_plans_created"] += 1
                audit[f"canonical_{winner}_wins"] += 1
                audit["canonical_planning_seconds"] += (
                    time.perf_counter() - planning_started
                )
        if len(_LABELED_2X2_BOUNDED_TREE_CACHE) >= 4096:
            _LABELED_2X2_BOUNDED_TREE_CACHE.clear()
        _LABELED_2X2_BOUNDED_TREE_CACHE[signature] = (
            actions,
            planned_peak,
            greedy_feasible,
        )
    else:
        actions, planned_peak, greedy_feasible = cached
        if audit is not None:
            audit["canonical_memory_cache_hits"] += 1

    restored_actions = _restore_bounded_2x2_action_labels(
        actions, canonical_to_original
    )
    result = _execute_bounded_2x2_tree(
        operands,
        labels,
        output_labels,
        restored_actions,
        description=description,
    )
    if newly_planned:
        _LABELED_2X2_PERSISTENT_CACHE_ENTRIES[digest] = {
            "signature": signature_json,
            "actions": actions,
            "planned_peak": int(planned_peak),
            "greedy_feasible": bool(greedy_feasible),
        }
        _LABELED_2X2_PERSISTENT_CACHE_DIRTY += 1
        if audit is not None:
            audit["persistent_entries_queued"] += 1
        if _LABELED_2X2_PERSISTENT_CACHE_DIRTY >= 8:
            flush_bounded_2x2_tree_cache()
    return result, int(planned_peak), not greedy_feasible


def _joint_halo_apply(
    left: torch.Tensor,
    right: torch.Tensor,
    probe: torch.Tensor,
) -> torch.Tensor:
    """Apply ``H = left @ right.T`` independently for every branch."""
    if left.ndim != 3 or right.ndim != 3 or probe.ndim != 3:
        raise ValueError("joint-halo apply expects three batched matrices")
    if (
        left.shape[0] != right.shape[0]
        or left.shape[0] != probe.shape[0]
        or left.shape[2] != right.shape[2]
        or right.shape[1] != probe.shape[1]
    ):
        raise ValueError("incompatible joint-halo apply dimensions")
    return left @ (right.transpose(-2, -1) @ probe)


def _joint_halo_adjoint_apply(
    left: torch.Tensor,
    right: torch.Tensor,
    probe: torch.Tensor,
) -> torch.Tensor:
    """Apply the conjugate transpose of ``H = left @ right.T``."""
    if left.ndim != 3 or right.ndim != 3 or probe.ndim != 3:
        raise ValueError("joint-halo adjoint apply expects batched matrices")
    if (
        left.shape[0] != right.shape[0]
        or left.shape[0] != probe.shape[0]
        or left.shape[2] != right.shape[2]
        or left.shape[1] != probe.shape[1]
    ):
        raise ValueError("incompatible joint-halo adjoint dimensions")
    return right.conj() @ (left.conj().transpose(-2, -1) @ probe)


def _randomized_joint_halo_factors(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    rank: int,
    oversample: int,
    power_iterations: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
    """Factor a batched halo operator without forming its dense matrix.

    ``left`` and ``right`` contain the two independently dressed halo sites,
    with their shared doubled bond as the last axis.  No branch is averaged or
    mixed with another branch.
    """
    if left.dtype != torch.complex128 or right.dtype != torch.complex128:
        raise ValueError("low-rank joint-halo currently requires complex128")
    if rank < 1 or oversample < 1 or power_iterations < 0:
        raise ValueError("invalid randomized joint-halo controls")
    batch, rows, shared = (int(value) for value in left.shape)
    if tuple(right.shape[:1]) != (batch,) or int(right.shape[2]) != shared:
        raise ValueError("joint-halo factors have incompatible shared bonds")
    cols = int(right.shape[1])
    sketch = min(rows, cols, rank + oversample)
    actual_rank = min(rank, rows, cols)
    generator = torch.Generator(device=left.device)
    generator.manual_seed(int(seed))
    real = torch.randn(
        cols, sketch, dtype=torch.float64, device=left.device, generator=generator
    )
    imag = torch.randn(
        cols, sketch, dtype=torch.float64, device=left.device, generator=generator
    )
    omega = torch.complex(real, imag).unsqueeze(0).expand(batch, -1, -1)
    q = torch.linalg.qr(_joint_halo_apply(left, right, omega), mode="reduced").Q
    for _ in range(power_iterations):
        q = torch.linalg.qr(
            _joint_halo_apply(left, right, _joint_halo_adjoint_apply(left, right, q)),
            mode="reduced",
        ).Q
    small = (q.conj().transpose(-2, -1) @ left) @ right.transpose(-2, -1)
    u_small, singular, vh = torch.linalg.svd(small, full_matrices=False)
    u = q @ u_small[..., :actual_rank]
    singular = singular[..., :actual_rank]
    vh = vh[..., :actual_rank, :]
    root = singular.sqrt()
    low_left = u * root.unsqueeze(-2)
    low_right = root.unsqueeze(-1) * vh

    audit_real = torch.randn(
        cols, 1, dtype=torch.float64, device=left.device, generator=generator
    )
    audit_imag = torch.randn(
        cols, 1, dtype=torch.float64, device=left.device, generator=generator
    )
    audit_probe = (
        torch.complex(audit_real, audit_imag).unsqueeze(0).expand(batch, -1, -1)
    )
    exact = _joint_halo_apply(left, right, audit_probe)
    approximate = low_left @ (low_right @ audit_probe)
    denominator = torch.linalg.vector_norm(exact.reshape(batch, -1), dim=1).clamp_min(
        1e-300
    )
    residual = (
        torch.linalg.vector_norm((exact - approximate).reshape(batch, -1), dim=1)
        / denominator
    )
    return (
        low_left,
        low_right,
        {
            "requested_rank": int(rank),
            "effective_rank": int(actual_rank),
            "sketch_rank": int(sketch),
            "probe_relative_residual_max": float(residual.max().item()),
        },
    )


def _joint_halo_half_matrix(
    ket: dict[int, torch.Tensor],
    bra: dict[int, torch.Tensor],
    messages: dict[tuple[int, str], torch.Tensor],
    site: int,
    *,
    toward_patch: str,
    toward_partner: str,
    plan: dict[str, Any],
    branch_slice: slice | None = None,
) -> torch.Tensor:
    """Dress one halo site, leaving patch and partner doubled legs open."""
    ket_value = ket[site] if branch_slice is None else ket[site][branch_slice]
    batch = int(ket_value.shape[0])
    ket_edges = {direction: index + 1 for index, direction in enumerate("UDLR")}
    bra_edges = {direction: index + 5 for index, direction in enumerate("UDLR")}
    physical = 9
    bra_value = bra[site]
    bra_batched = bra_value.ndim == ket[site].ndim
    if bra_batched:
        bra_value = bra_value if branch_slice is None else bra_value[branch_slice]
        if int(bra_value.shape[0]) != batch:
            raise ValueError("batched halo bra does not match ket batch")
    operands = [ket_value, bra_value.conj()]
    labels = [
        [0, *(ket_edges[direction] for direction in "UDLR"), physical],
        ([0] if bra_batched else [])
        + [*(bra_edges[direction] for direction in "UDLR"), physical],
    ]
    for direction, other in plan["neighbors"][site].items():
        if direction in (toward_patch, toward_partner):
            continue
        message = messages[(other, OPPOSITE[direction])]
        operands.append(message if branch_slice is None else message[branch_slice])
        labels.append([0, ket_edges[direction], bra_edges[direction]])
    output = [
        0,
        ket_edges[toward_patch],
        bra_edges[toward_patch],
        ket_edges[toward_partner],
        bra_edges[toward_partner],
    ]
    value = _contract_labeled_pairwise(
        operands,
        labels,
        output,
        batch_label=0,
        memory_limit=int(plan["memory_limit"]),
        trace_labels={physical},
        description="joint-halo half contraction",
    )
    return value.reshape(
        batch,
        int(value.shape[1]) * int(value.shape[2]),
        int(value.shape[3]) * int(value.shape[4]),
    )


_JOINT_HALO_HALF_WORKSPACE_BYTES = 1 << 30


def _joint_halo_half_elements_per_branch(
    ket: dict[int, torch.Tensor],
    bra: dict[int, torch.Tensor],
    site: int,
    *,
    toward_patch: str,
    toward_partner: str,
) -> int:
    """Return the dense half-matrix output size for one source branch."""
    ket_value = ket[site]
    bra_value = bra[site]
    bra_offset = 1 if bra_value.ndim == ket_value.ndim else 0

    def doubled_leg(direction: str) -> int:
        axis = DIR_TO_AXIS[direction]
        return int(ket_value.shape[1 + axis]) * int(bra_value.shape[bra_offset + axis])

    return doubled_leg(toward_patch) * doubled_leg(toward_partner)


def _low_rank_joint_halo_factors(
    ket: dict[int, torch.Tensor],
    bra: dict[int, torch.Tensor],
    messages: dict[tuple[int, str], torch.Tensor],
    first: int,
    second: int,
    *,
    direction: str,
    internal: str,
    plan: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return two truncated patch-facing factors for one halo side."""
    toward_patch = OPPOSITE[direction]
    reverse_internal = OPPOSITE[internal]
    batch = int(ket[first].shape[0])
    half_elements = _joint_halo_half_elements_per_branch(
        ket, bra, first, toward_patch=toward_patch, toward_partner=internal
    ) + _joint_halo_half_elements_per_branch(
        ket, bra, second, toward_patch=toward_patch, toward_partner=reverse_internal
    )
    half_bytes = half_elements * int(ket[first].element_size())
    branch_step = min(
        batch,
        max(1, _JOINT_HALO_HALF_WORKSPACE_BYTES // max(1, half_bytes)),
    )
    requested_rank = int(plan["joint_halo_rank"])
    aggregate = plan["joint_halo_rank_audit"]
    low_left_parts = []
    low_right_parts = []
    direction_seed = "UDLR".index(direction)
    for start in range(0, batch, branch_step):
        stop = min(start + branch_step, batch)
        branch_slice = slice(start, stop)
        left = _joint_halo_half_matrix(
            ket,
            bra,
            messages,
            first,
            toward_patch=toward_patch,
            toward_partner=internal,
            plan=plan,
            branch_slice=branch_slice,
        )
        right = _joint_halo_half_matrix(
            ket,
            bra,
            messages,
            second,
            toward_patch=toward_patch,
            toward_partner=reverse_internal,
            plan=plan,
            branch_slice=branch_slice,
        )
        shared_rank = int(left.shape[2])
        if shared_rank <= requested_rank:
            chunk_size = int(left.shape[0])
            aggregate["factorization_count"] += chunk_size
            aggregate["exact_factor_bypass_count"] += chunk_size
            effective = shared_rank
            low_left = left
            low_right = right.transpose(-2, -1)
        else:
            low_left, low_right, audit = _randomized_joint_halo_factors(
                left,
                right,
                rank=requested_rank,
                oversample=int(plan["joint_halo_oversample"]),
                power_iterations=int(plan["joint_halo_power_iterations"]),
                seed=(int(plan["joint_halo_random_seed"]) + direction_seed),
            )
            effective = int(audit["effective_rank"])
            aggregate["factorization_count"] += int(low_left.shape[0])
            aggregate["max_probe_relative_residual"] = max(
                float(aggregate["max_probe_relative_residual"]),
                float(audit["probe_relative_residual_max"]),
            )
        aggregate["effective_rank_min"] = (
            effective
            if aggregate["effective_rank_min"] is None
            else min(int(aggregate["effective_rank_min"]), effective)
        )
        aggregate["effective_rank_max"] = (
            effective
            if aggregate["effective_rank_max"] is None
            else max(int(aggregate["effective_rank_max"]), effective)
        )
        low_left_parts.append(low_left)
        low_right_parts.append(low_right)
        del left, right, low_left, low_right
    if len(low_left_parts) == 1:
        return low_left_parts[0], low_right_parts[0]
    return (
        torch.cat(low_left_parts, dim=0),
        torch.cat(low_right_parts, dim=0),
    )


def _matrix_free_2x2_network(
    ket: dict[int, torch.Tensor],
    bra: dict[int, torch.Tensor],
    messages: dict[tuple[int, str], torch.Tensor],
    patch: tuple[int, ...],
    support: tuple[int, ...],
    cluster_support: tuple[int, ...],
    plan: dict[str, Any],
) -> tuple[
    list[torch.Tensor],
    list[list[int]],
    list[int],
    set[int],
    int,
    list[tuple[int, int]],
]:
    """Build the exact labeled 2x2 joint-halo tensor network."""
    _emit_stage_profile_event(
        "matrix_free_network_construction",
        "begin",
        batch_rows=int(next(iter(ket.values())).shape[0]),
    )
    if len(patch) != 4:
        raise ValueError("matrix-free readout is restricted to 2x2 patches")
    B = int(next(iter(ket.values())).shape[0])
    batch_label = 0
    next_label = 1

    def fresh_pair() -> tuple[int, int]:
        nonlocal next_label
        pair = (next_label, next_label + 1)
        next_label += 2
        return pair

    patch_set = set(patch)
    virtual: dict[tuple[int, str], tuple[int, int]] = {}
    for site in patch:
        for direction in "UDLR":
            key = (site, direction)
            if key in virtual:
                continue
            pair = fresh_pair()
            virtual[key] = pair
            other = plan["neighbors"][site].get(direction)
            if other in patch_set:
                virtual[(other, OPPOSITE[direction])] = pair

    operands: list[torch.Tensor] = []
    labels: list[list[int]] = []
    operand_positions: list[tuple[int, int]] = []
    trace_labels: set[int] = set()
    physical_ket: dict[int, int] = {}
    physical_bra: dict[int, int] = {}

    def append_site(
        site: int,
        edge_labels: dict[str, tuple[int, int]],
        *,
        keep_physical: bool,
    ) -> tuple[int, int]:
        nonlocal next_label
        ket_physical = next_label
        next_label += 1
        if keep_physical:
            bra_physical = next_label
            next_label += 1
        else:
            bra_physical = ket_physical
            trace_labels.add(ket_physical)
        ket_edges = [edge_labels[direction][0] for direction in "UDLR"]
        bra_edges = [edge_labels[direction][1] for direction in "UDLR"]
        bra_value = bra[site].conj()
        batched_bra = bra_value.ndim == ket[site].ndim
        if batched_bra and int(bra_value.shape[0]) != B:
            raise ValueError("batched cluster bra does not match ket batch")
        operands.extend((ket[site], bra_value))
        labels.extend(
            (
                [batch_label, *ket_edges, ket_physical],
                ([batch_label] if batched_bra else []) + [*bra_edges, bra_physical],
            )
        )
        position = divmod(site, int(plan["width"]))
        operand_positions.extend((position, position))
        return ket_physical, bra_physical

    support_set = set(support)
    for site in patch:
        edges = {direction: virtual[(site, direction)] for direction in "UDLR"}
        physical_ket[site], physical_bra[site] = append_site(
            site, edges, keep_physical=site in support_set
        )

    joint_edges: set[tuple[int, str]] = set()
    if plan["readout_boundary_mode"] == "joint_halo":
        for direction in "UDLR":
            side_sites = _joint_side_patch_sites(
                patch, cluster_support, direction, plan
            )
            halo = tuple(plan["neighbors"][site].get(direction) for site in side_sites)
            if halo[0] is None or halo[1] is None:
                continue
            first, second = int(halo[0]), int(halo[1])
            internal = next(
                (
                    side
                    for side, other in plan["neighbors"][first].items()
                    if other == second
                ),
                None,
            )
            if internal is None:
                raise RuntimeError(f"non-adjacent halo sites {halo}")
            reverse_internal = OPPOSITE[internal]
            toward_patch = OPPOSITE[direction]
            if plan.get("joint_halo_rank") is not None:
                low_left, low_right = _low_rank_joint_halo_factors(
                    ket,
                    bra,
                    messages,
                    first,
                    second,
                    direction=direction,
                    internal=internal,
                    plan=plan,
                )
                rank_label = next_label
                next_label += 1
                for local, (halo_site, factor) in enumerate(
                    (
                        (first, low_left),
                        (second, low_right.transpose(-2, -1)),
                    )
                ):
                    patch_site = side_sites[local]
                    axis = DIR_TO_AXIS[toward_patch]
                    ket_leg = int(ket[halo_site].shape[1 + axis])
                    bra_offset = 1 if bra[halo_site].ndim == ket[halo_site].ndim else 0
                    bra_leg = int(bra[halo_site].shape[bra_offset + axis])
                    factor = factor.reshape(B, ket_leg, bra_leg, -1)
                    patch_ket, patch_bra = virtual[(patch_site, direction)]
                    operands.append(factor)
                    labels.append([batch_label, patch_ket, patch_bra, rank_label])
                    operand_positions.append(divmod(halo_site, int(plan["width"])))
                    joint_edges.add((patch_site, direction))
                continue
            shared = fresh_pair()
            for local, halo_site in enumerate((first, second)):
                patch_site = side_sites[local]
                partner_direction = internal if local == 0 else reverse_internal
                halo_edges = {}
                for side in "UDLR":
                    if side == toward_patch:
                        halo_edges[side] = virtual[(patch_site, direction)]
                    elif side == partner_direction:
                        halo_edges[side] = shared
                    else:
                        halo_edges[side] = fresh_pair()
                append_site(halo_site, halo_edges, keep_physical=False)
                for side, other in plan["neighbors"][halo_site].items():
                    if side in (toward_patch, partner_direction):
                        continue
                    edge_ket, edge_bra = halo_edges[side]
                    operands.append(messages[(other, OPPOSITE[side])])
                    labels.append([batch_label, edge_ket, edge_bra])
                    operand_positions.append(divmod(halo_site, int(plan["width"])))
                joint_edges.add((patch_site, direction))

    for site in patch:
        for direction, other in plan["neighbors"][site].items():
            if other in patch_set or (site, direction) in joint_edges:
                continue
            edge_ket, edge_bra = virtual[(site, direction)]
            operands.append(messages[(other, OPPOSITE[direction])])
            labels.append([batch_label, edge_ket, edge_bra])
            operand_positions.append(divmod(site, int(plan["width"])))

    output_labels = [batch_label]
    output_labels.extend(physical_ket[site] for site in support)
    output_labels.extend(physical_bra[site] for site in support)
    if len(operand_positions) != len(operands):
        raise RuntimeError("matrix-free operand position metadata mismatch")
    _emit_stage_profile_event(
        "matrix_free_network_construction",
        "end",
        batch_rows=B,
        operand_count=len(operands),
    )
    return (
        operands,
        labels,
        output_labels,
        trace_labels,
        B,
        operand_positions,
    )


def _apply_matrix_free_operators(
    raw: torch.Tensor,
    operators: torch.Tensor | None,
) -> torch.Tensor:
    if operators is None:
        return raw
    _emit_stage_profile_event(
        "target_mode_contraction",
        "begin",
        batch_rows=int(raw.shape[0]),
        operator_rank=int(operators.ndim),
    )
    if operators.ndim == 3:
        result = torch.einsum("vab,zba->zv", operators, raw)
    elif operators.ndim == 4:
        result = torch.einsum("zvab,zba->zv", operators, raw)
    else:
        raise ValueError("matrix-free operators must have shape [V,d,d] or [B,V,d,d]")
    _emit_stage_profile_event(
        "target_mode_contraction",
        "end",
        batch_rows=int(raw.shape[0]),
        operator_rank=int(operators.ndim),
    )
    return result


def _matrix_free_2x2_transition(
    ket: dict[int, torch.Tensor],
    bra: dict[int, torch.Tensor],
    messages: dict[tuple[int, str], torch.Tensor],
    patch: tuple[int, ...],
    support: tuple[int, ...],
    cluster_support: tuple[int, ...],
    plan: dict[str, Any],
) -> torch.Tensor:
    """Exact 2x2 contraction without materializing dense joint-halo tensors."""
    (operands, labels, output_labels, trace_labels, B, operand_positions) = (
        _matrix_free_2x2_network(
            ket, bra, messages, patch, support, cluster_support, plan
        )
    )
    if plan.get("joint_halo_rank") is None:
        raw = _contract_labeled_pairwise(
            operands,
            labels,
            output_labels,
            batch_label=0,
            memory_limit=int(plan["memory_limit"]),
            trace_labels=trace_labels,
            description="matrix-free 2x2 readout contraction",
        )
    else:
        raw, planned_peak, used_fallback = _contract_labeled_2x2_bounded(
            operands,
            labels,
            output_labels,
            batch_label=0,
            memory_limit=int(plan["memory_limit"]),
            trace_labels=trace_labels,
            operand_positions=operand_positions,
            description="matrix-free 2x2 readout contraction",
            audit=plan["bounded_2x2_path_audit"],
        )
        audit = plan["bounded_2x2_path_audit"]
        audit["canonical_tree_contraction_count"] += B
        audit["max_planned_intermediate_elements"] = max(
            int(audit["max_planned_intermediate_elements"]),
            int(planned_peak),
        )
        if used_fallback:
            audit["fallback_contraction_count"] += B
    dimension = 2 ** len(support)
    return raw.reshape(B, dimension, dimension)


def _cluster_contract(
    ket: dict[int, torch.Tensor],
    bra: dict[int, torch.Tensor],
    messages: dict[tuple[int, str], torch.Tensor],
    patch: tuple[int, ...],
    plan: dict[str, Any],
    *,
    support: tuple[int, ...] = (),
    out_edge: tuple[int, str] | None = None,
    joint_boundary: dict[str, tuple[tuple[int, int], torch.Tensor]] | None = None,
    operators: torch.Tensor | None = None,
) -> torch.Tensor:
    """Contract a local cluster, optionally leaving one boundary leg open."""
    if out_edge is not None and out_edge[0] not in patch:
        raise ValueError(f"outgoing edge {out_edge} is outside the patch")
    if out_edge is not None and operators is not None:
        raise ValueError("operator projection cannot leave a boundary leg open")

    # Assign one label to each doubled virtual and physical edge.
    label_pool = iter("abcdefghijklmnopqrstuvwxy")
    virtual_labels: dict[tuple[int, str], str] = {}
    patch_set = set(patch)
    for site in patch:
        for direction in "UDLR":
            key = (site, direction)
            if key in virtual_labels:
                continue
            label = next(label_pool)
            virtual_labels[key] = label
            other = plan["neighbors"][site].get(direction)
            if other in patch_set:
                virtual_labels[(other, OPPOSITE[direction])] = label
    labels = {
        site: tuple(virtual_labels[(site, direction)] for direction in "UDLR")
        + (next(label_pool),)
        for site in patch
    }
    joint_edges = set()
    if joint_boundary is not None:
        joint_edges = {
            (site, direction)
            for direction, (sites, _) in joint_boundary.items()
            for site in sites
        }
    fragments, operands = [], []
    support_set = set(support)
    for site in patch:
        virtual = labels[site][:4]
        physical = labels[site][4]
        fragments.append("z " + " ".join((*virtual, physical)))
        operands.append(ket[site])
        bra_physical = physical.upper() if site in support_set else physical
        bra_value = bra[site].conj()
        batched_bra = bra_value.ndim == ket[site].ndim
        if batched_bra and int(bra_value.shape[0]) != int(ket[site].shape[0]):
            raise ValueError("batched cluster bra does not match the ket batch")
        fragments.append(
            (
                ("z " if batched_bra else "")
                + " ".join((*[label.upper() for label in virtual], bra_physical))
            )
        )
        operands.append(bra_value)
        for direction, label in zip("UDLR", virtual):
            other = plan["neighbors"][site].get(direction)
            if other is not None and other not in patch:
                if out_edge == (site, direction):
                    continue
                if (site, direction) in joint_edges:
                    continue
                fragments.append(f"z {label} {label.upper()}")
                operands.append(messages[(other, OPPOSITE[direction])])

    if joint_boundary is not None:
        for direction, (side_sites, value) in joint_boundary.items():
            side_labels = [labels[site][DIR_TO_AXIS[direction]] for site in side_sites]
            fragments.append(
                "z "
                + " ".join(
                    label for local in side_labels for label in (local, local.upper())
                )
            )
            operands.append(value)

    physical = {site: labels[site][4] for site in patch}
    output = [
        "z",
        *(physical[site] for site in support),
        *(physical[site].upper() for site in support),
    ]
    if operators is not None:
        if not support:
            raise ValueError("operator projection requires a physical support")
        if "v" in {label for site in patch for label in labels[site]}:
            raise RuntimeError("fused operator label collides with cluster labels")
        dimension = 2 ** len(support)
        if operators.ndim not in (3, 4) or tuple(
            int(x) for x in operators.shape[-2:]
        ) != (dimension, dimension):
            raise ValueError("fused operators must have shape [V,d,d] or [B,V,d,d]")
        batched_operators = operators.ndim == 4
        if batched_operators and int(operators.shape[0]) != int(
            next(iter(ket.values())).shape[0]
        ):
            raise ValueError("batched fused operators do not match the ket batch")
        prefix = tuple(int(x) for x in operators.shape[:-2])
        operators = operators.reshape(
            *prefix,
            *([2] * len(support)),
            *([2] * len(support)),
        )
        fragments.append(
            (
                ("z " if batched_operators else "")
                + "v "
                + " ".join(
                    [physical[site].upper() for site in support]
                    + [physical[site] for site in support]
                )
            )
        )
        operands.append(operators)
        output = ["z", "v"]
    if out_edge is not None:
        site, direction = out_edge
        output.extend(
            (
                labels[site][DIR_TO_AXIS[direction]],
                labels[site][DIR_TO_AXIS[direction]].upper(),
            )
        )
    raw = _einsum(
        ", ".join(fragments) + " -> " + " ".join(output),
        *operands,
        memory_limit=plan["memory_limit"],
    )
    if out_edge is not None:
        return raw
    if operators is not None:
        return raw
    dimension = 2 ** len(support)
    return raw.reshape(-1, dimension, dimension)


def _support_only_transition(
    ket: dict[int, torch.Tensor],
    bra: dict[int, torch.Tensor],
    messages: dict[tuple[int, str], torch.Tensor],
    support: tuple[int, ...],
    plan: dict[str, Any],
) -> torch.Tensor:
    """Original BP belief on exactly the physical target support."""
    if len(support) == 1:
        site = support[0]
        return _contract_site(
            ket,
            bra,
            site,
            messages,
            plan,
            keep_phys=True,
        ).reshape(-1, 2, 2)
    if len(support) != 2:
        raise ValueError(f"unsupported target support {support}")
    first_site, second_site = support
    direction = next(
        (
            side
            for side, other in plan["neighbors"][first_site].items()
            if other == second_site
        ),
        None,
    )
    if direction is None:
        raise ValueError(f"support {support} is not nearest-neighbour")
    reverse = OPPOSITE[direction]
    first = _contract_site(
        ket, bra, first_site, messages, plan, out_dir=direction, keep_phys=True
    )
    second = _contract_site(
        ket, bra, second_site, messages, plan, out_dir=reverse, keep_phys=True
    )
    raw = torch.einsum("zpPaA,zqQaA->zpPqQ", first, second)
    return raw.permute(0, 1, 3, 2, 4).contiguous().reshape(-1, 4, 4)


def _gen_local_gloops(
    support: tuple[int, ...],
    plan: dict[str, Any],
) -> tuple[frozenset[int], ...]:
    """Generate local generalized loops without constructing a Quimb TN.

    This is the square-lattice specialization of Quimb's
    ``_gen_gloops_single``.  All non-target sites must have degree at least
    two inside the region; target sites may dangle for ``*dangle`` growth.
    Only the resulting small regions are later handed to Quimb/Cotengra for
    contraction -- the full lattice is never converted or contracted.
    """
    support = tuple(int(site) for site in support)
    grow_from = str(plan["gloop_grow_from"])
    max_size = int(plan["gloop_size"])
    key = (
        int(plan["width"]),
        int(plan["length"]),
        support,
        grow_from,
        max_size,
    )
    cached = plan.setdefault("gloop_regions", {}).get(key)
    if cached is not None:
        return cached

    if grow_from in {"all", "alldangle"}:
        queue = [frozenset(support)]
    else:
        queue = [frozenset((site,)) for site in support]
    dangle_sites = set(support) if "dangle" in grow_from else set()
    seen = set(queue)
    valid_regions: set[frozenset[int]] = set()
    cursor = 0
    while cursor < len(queue):
        region = queue[cursor]
        cursor += 1
        degrees = {
            site: sum(
                int(other in region) for other in plan["neighbors"][site].values()
            )
            for site in region
        }
        valid = all(
            site in dangle_sites or degree >= 2 for site, degree in degrees.items()
        )
        if valid:
            # ``any`` growth in Quimb treats unselected target tensors as
            # readout extras.  Add them here so every contracted region can
            # expose the complete requested physical support.
            completed = region | set(support)
            if len(completed) <= max_size:
                valid_regions.add(frozenset(completed))

        if len(region) >= max_size:
            continue
        next_sites = {
            other
            for site in region
            for other in plan["neighbors"][site].values()
            if other not in region
        }
        for other in sorted(next_sites):
            expanded = region | {other}
            if expanded not in seen:
                seen.add(expanded)
                queue.append(expanded)

    if not valid_regions:
        raise RuntimeError(
            f"no generalized-loop region found for support={support}, "
            f"size={max_size}, grow_from={grow_from}"
        )
    result = tuple(
        sorted(valid_regions, key=lambda region: (len(region), tuple(sorted(region))))
    )
    plan["gloop_regions"][key] = result
    return result


def _gloop_region_counts(
    support: tuple[int, ...],
    plan: dict[str, Any],
) -> tuple[tuple[frozenset[int], int], ...]:
    """Return Quimb's exact local-gloop regions and Möbius factors."""
    cache_key = (
        int(plan["width"]),
        int(plan["length"]),
        tuple(int(site) for site in support),
        str(plan["gloop_grow_from"]),
        int(plan["gloop_size"]),
    )
    cached = _GLOOP_REGION_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        import quimb.tensor as qtn
        from quimb.tensor.belief_propagation.regions import gen_region_counts
    except ImportError as error:
        raise RuntimeError(
            "local_readout_method='gloop' requires quimb and cotengra"
        ) from error

    topology = plan.get("gloop_topology")
    if topology is None:
        width = int(plan["width"])
        length = int(plan["length"])
        arrays = []
        site_coordinates = {}
        for qrow in range(length):
            core_row = length - 1 - qrow
            line = []
            for col in range(width):
                site = core_row * width + col
                full = torch.ones((1, 1, 1, 1, 1), dtype=torch.float64)
                missing = {
                    "u": qrow + 1 == length,
                    "r": col + 1 == width,
                    "d": qrow == 0,
                    "l": col == 0,
                    "p": False,
                }
                line.append(
                    full[
                        tuple(0 if missing[label] else slice(None) for label in "urdlp")
                    ]
                )
                site_coordinates[site] = (qrow, col)
            arrays.append(line)
        topology_tn = qtn.PEPS(arrays, shape="urdlp")
        site_to_tid = {
            site: next(
                iter(
                    topology_tn._get_tids_from_tags(
                        topology_tn.site_tag(coordinate), "all"
                    )
                )
            )
            for site, coordinate in site_coordinates.items()
        }
        tid_to_site = {tid: site for site, tid in site_to_tid.items()}
        topology = {
            "tn": topology_tn,
            "site_to_tid": site_to_tid,
            "tid_to_site": tid_to_site,
        }
        plan["gloop_topology"] = topology

    support_tids = tuple(topology["site_to_tid"][int(site)] for site in support)
    gloops = topology["tn"].get_local_gloops(
        tids=support_tids,
        gloops=int(plan["gloop_size"]),
        grow_from=str(plan["gloop_grow_from"]),
        strict_size=False,
    )
    result = tuple(
        (
            frozenset(topology["tid_to_site"][tid] for tid in region),
            int(count),
        )
        for region, count in gen_region_counts(gloops)
    )
    if not result:
        raise RuntimeError(f"Quimb produced no gloop regions for support={support}")
    if len(_GLOOP_REGION_CACHE) >= 4096:
        _GLOOP_REGION_CACHE.clear()
    _GLOOP_REGION_CACHE[cache_key] = result
    return result


def _quimb_contract_labeled(
    operands: list[torch.Tensor],
    inputs: list[tuple[int, ...]],
    output: tuple[int, ...],
    plan: dict[str, Any],
) -> torch.Tensor:
    """Contract one small labeled cluster with a cached Cotengra expression."""
    if plan.get("gloop_peak_budget_gib") is not None:
        from residual_tn.backend.bounded_gloop import contract

        return contract(operands, inputs, output, plan, _GLOOP_CONTRACT_EXPR_CACHE)
    try:
        from quimb.tensor.contraction import (
            array_contract_expression,
            array_contract_tree,
        )
    except ImportError as error:
        raise RuntimeError(
            "local_readout_method='gloop' requires quimb and cotengra; "
            "the streamed PEPS/BP engine itself remains independent"
        ) from error

    shapes = tuple(
        tuple(int(dimension) for dimension in operand.shape) for operand in operands
    )
    key = (
        tuple(inputs),
        tuple(output),
        shapes,
        str(plan["gloop_optimize"]),
        plan.get("gloop_memory_target_gib"),
        int(plan.get("gloop_memory_search_repeats", 32)),
        int(plan.get("gloop_memory_max_slices", 64)),
        int(operands[0].element_size()),
    )
    expression = _GLOOP_CONTRACT_EXPR_CACHE.get(key)
    audit = plan["gloop_audit"]
    if expression is None:
        audit["expression_cache_misses"] += 1
        build_started = time.perf_counter()
        memory_target_gib = plan.get("gloop_memory_target_gib")
        if memory_target_gib is None:
            expression = array_contract_expression(
                inputs=tuple(inputs),
                output=tuple(output),
                shapes=shapes,
                optimize=plan["gloop_optimize"],
            )
        else:
            try:
                import cotengra
            except ImportError as error:
                raise RuntimeError(
                    "memory-aware gloop planning requires cotengra"
                ) from error

            element_size = int(operands[0].element_size())
            target_elements = max(
                1,
                int(float(memory_target_gib) * 1024.0**3 / element_size),
            )
            tree = array_contract_tree(
                inputs=tuple(inputs),
                output=tuple(output),
                shapes=shapes,
                optimize=plan["gloop_optimize"],
            )
            original_max = int(tree.max_size())
            final_max = original_max
            audit["memory_planned_expression_count"] += 1
            audit["memory_max_original_intermediate_elements"] = max(
                int(audit["memory_max_original_intermediate_elements"]),
                original_max,
            )

            if original_max > target_elements:
                optimizer = plan.get("gloop_memory_optimizer")
                if optimizer is None:
                    optimizer = cotengra.ReusableHyperOptimizer(
                        methods=["greedy", "random-greedy"],
                        minimize="size",
                        max_repeats=int(plan["gloop_memory_search_repeats"]),
                        parallel=False,
                        progbar=False,
                        hash_method="b",
                    )
                    plan["gloop_memory_optimizer"] = optimizer
                memory_tree = array_contract_tree(
                    inputs=tuple(inputs),
                    output=tuple(output),
                    shapes=shapes,
                    optimize=optimizer,
                )
                memory_max = int(memory_tree.max_size())
                audit["memory_researched_expression_count"] += 1
                if memory_max < final_max:
                    tree = memory_tree
                    final_max = memory_max

            if final_max > target_elements:
                sliced_tree = tree.slice(
                    target_size=target_elements,
                    minimize="size",
                    allow_outer=False,
                    max_repeats=32,
                    inplace=False,
                )
                slice_count = int(sliced_tree.nslices)
                max_slices = int(plan["gloop_memory_max_slices"])
                if slice_count > max_slices:
                    raise RuntimeError(
                        "memory-aware gloop path requires "
                        f"{slice_count} exact slices, above configured "
                        f"gloop_memory_max_slices={max_slices}; "
                        f"original_max_gib="
                        f"{original_max * element_size / 1024.0**3:.3f}, "
                        f"researched_max_gib="
                        f"{final_max * element_size / 1024.0**3:.3f}, "
                        f"target_gib={float(memory_target_gib):.3f}"
                    )
                tree = sliced_tree
                final_max = int(tree.max_size())
                audit["memory_sliced_expression_count"] += 1
                audit["memory_max_slice_count"] = max(
                    int(audit["memory_max_slice_count"]), slice_count
                )
            else:
                slice_count = 1

            audit["memory_max_final_intermediate_elements"] = max(
                int(audit["memory_max_final_intermediate_elements"]),
                final_max,
            )
            if original_max > target_elements or slice_count > 1:
                print(
                    "[gloop-path-memory] "
                    f"original_gib="
                    f"{original_max * element_size / 1024.0**3:.3f} "
                    f"final_gib="
                    f"{final_max * element_size / 1024.0**3:.3f} "
                    f"target_gib={float(memory_target_gib):.3f} "
                    f"slices={slice_count}",
                    flush=True,
                )
            expression = array_contract_expression(
                inputs=tuple(inputs),
                output=tuple(output),
                shapes=shapes,
                optimize=tree,
            )
        build_seconds = time.perf_counter() - build_started
        audit["expression_build_seconds"] += build_seconds
        audit["max_expression_build_seconds"] = max(
            float(audit["max_expression_build_seconds"]), build_seconds
        )
        limit = int(plan["gloop_expression_cache_size"])
        if len(_GLOOP_CONTRACT_EXPR_CACHE) >= limit:
            # Evict only the least-recently-used expression. Clearing the
            # entire cache at capacity makes an all-site/all-layer job rebuild
            # tens of thousands of otherwise reusable contraction paths.
            oldest_key = next(iter(_GLOOP_CONTRACT_EXPR_CACHE))
            _GLOOP_CONTRACT_EXPR_CACHE.pop(oldest_key)
            audit["expression_cache_evictions"] += 1
        _GLOOP_CONTRACT_EXPR_CACHE[key] = expression
    else:
        audit["expression_cache_hits"] += 1
        # Python dicts preserve insertion order, so refresh the hit to make
        # the single-entry eviction above a bounded LRU policy.
        _GLOOP_CONTRACT_EXPR_CACHE.pop(key)
        _GLOOP_CONTRACT_EXPR_CACHE[key] = expression
    return expression(*operands)


def _gloop_region_transition(
    ket: dict[int, torch.Tensor],
    bra: dict[int, torch.Tensor],
    messages: dict[tuple[int, str], torch.Tensor],
    region: frozenset[int],
    support: tuple[int, ...],
    plan: dict[str, Any],
) -> torch.Tensor:
    """Contract one BP-closed generalized-loop region."""
    region_set = set(region)
    batch_label = 0
    next_label = 1

    def fresh() -> int:
        nonlocal next_label
        label = next_label
        next_label += 1
        return label

    ket_virtual: dict[tuple[int, str], int] = {}
    bra_virtual: dict[tuple[int, str], int] = {}
    for site in sorted(region):
        for direction in "UDLR":
            key = (site, direction)
            if key in ket_virtual:
                continue
            other = plan["neighbors"][site].get(direction)
            reverse_key = (other, OPPOSITE[direction]) if other in region_set else None
            ket_label = fresh()
            bra_label = fresh()
            ket_virtual[key] = ket_label
            bra_virtual[key] = bra_label
            if reverse_key is not None:
                ket_virtual[reverse_key] = ket_label
                bra_virtual[reverse_key] = bra_label

    support_set = set(support)
    ket_physical = {site: fresh() for site in support}
    bra_physical = {site: fresh() for site in support}
    traced_physical = {site: fresh() for site in region if site not in support_set}
    operands: list[torch.Tensor] = []
    inputs: list[tuple[int, ...]] = []
    for site in sorted(region):
        physical_ket = (
            ket_physical[site] if site in support_set else traced_physical[site]
        )
        physical_bra = (
            bra_physical[site] if site in support_set else traced_physical[site]
        )
        ket_labels = tuple(ket_virtual[(site, direction)] for direction in "UDLR") + (
            physical_ket,
        )
        ket_value = ket[site]
        if ket_value.ndim == 6:
            ket_labels = (batch_label, *ket_labels)
        operands.append(ket_value)
        inputs.append(tuple(ket_labels))

        bra_labels = tuple(bra_virtual[(site, direction)] for direction in "UDLR") + (
            physical_bra,
        )
        bra_value = bra[site].conj()
        if bra_value.ndim == 6:
            bra_labels = (batch_label, *bra_labels)
        operands.append(bra_value)
        inputs.append(tuple(bra_labels))

        for direction, other in plan["neighbors"][site].items():
            if other in region_set:
                continue
            message = messages[(other, OPPOSITE[direction])]
            message_labels = (
                ket_virtual[(site, direction)],
                bra_virtual[(site, direction)],
            )
            if message.ndim == 3:
                message_labels = (batch_label, *message_labels)
            operands.append(message)
            inputs.append(tuple(message_labels))

    output = (
        batch_label,
        *(ket_physical[site] for site in support),
        *(bra_physical[site] for site in support),
    )
    raw = _quimb_contract_labeled(operands, inputs, output, plan)
    dimension = 2 ** len(support)
    return raw.reshape(-1, dimension, dimension)


def _gloop_transition_unchecked(
    ket: dict[int, torch.Tensor],
    bra: dict[int, torch.Tensor],
    messages: dict[tuple[int, str], torch.Tensor],
    support: tuple[int, ...],
    plan: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Enqueue a normalized gloop transition with a deferred trace audit."""
    audit = plan["gloop_audit"]
    audit["calls"] += 1
    terms = []
    invalid_trace = torch.zeros(
        (), dtype=torch.bool, device=next(iter(ket.values())).device
    )
    for region, count in _gloop_region_counts(support, plan):
        raw = _gloop_region_transition(ket, bra, messages, region, support, plan)
        trace = raw.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        # Keep this audit on device. A Python bool here synchronizes after
        # every region and prevents independent site streams from overlapping.
        invalid_trace = (
            invalid_trace | ((~torch.isfinite(trace)) | (trace.abs() <= 1e-30)).any()
        )
        normalized = raw / trace.reshape(-1, 1, 1)
        terms.append((normalized, count))
        audit["region_contractions"] += 1
        audit["max_region_sites"] = max(int(audit["max_region_sites"]), len(region))
        size_key = str(len(region))
        audit["region_count_by_size"][size_key] = (
            int(audit["region_count_by_size"].get(size_key, 0)) + 1
        )

    if plan["gloop_combine"] == "sum":
        result = sum(count * value for value, count in terms)
    else:
        result = torch.ones_like(terms[0][0])
        for value, count in terms:
            result = result * value.pow(count)
    trace = result.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    invalid_trace = (
        invalid_trace | ((~torch.isfinite(trace)) | (trace.abs() <= 1e-30)).any()
    )
    return result / trace.reshape(-1, 1, 1), invalid_trace


def _gloop_transition(
    ket: dict[int, torch.Tensor],
    bra: dict[int, torch.Tensor],
    messages: dict[tuple[int, str], torch.Tensor],
    support: tuple[int, ...],
    plan: dict[str, Any],
) -> torch.Tensor:
    """Generalized-loop local transition with per-region normalization."""
    result, invalid_trace = _gloop_transition_unchecked(
        ket, bra, messages, support, plan
    )
    if bool(invalid_trace):
        raise RuntimeError(
            f"zero/non-finite generalized-loop trace for support={support}"
        )
    return result


def _raw_transition(
    ket: dict[int, torch.Tensor],
    bra: dict[int, torch.Tensor],
    messages: dict[tuple[int, str], torch.Tensor],
    support: tuple[int, ...],
    plan: dict[str, Any],
    *,
    readout_support: tuple[int, ...] | None = None,
    operators: torch.Tensor | None = None,
) -> torch.Tensor:
    """Contract the local readout cluster with the configured boundary."""
    if len(support) not in (1, 2):
        raise ValueError(f"unsupported target support {support}")
    if plan.get("local_readout_method", "cluster") == "gloop":
        if operators is not None:
            raw = _gloop_transition(ket, bra, messages, support, plan)
            if operators.ndim == 3:
                return torch.einsum("vab,zba->zv", operators, raw)
            if operators.ndim == 4:
                return torch.einsum("zvab,zba->zv", operators, raw)
            raise ValueError("gloop operators must have shape [V,d,d] or [B,V,d,d]")
        return _gloop_transition(ket, bra, messages, support, plan)
    if plan["readout_block"] == "support":
        if operators is not None:
            raise ValueError(
                "fused operator readout is not implemented for support-only"
            )
        return _support_only_transition(ket, bra, messages, support, plan)
    cluster_support = readout_support or support
    patch = _select_readout_patch(cluster_support, plan)
    if (
        plan.get("readout_contraction_mode", "einsum") == "matrix_free"
        and len(patch) == 4
    ):
        raw = _matrix_free_2x2_transition(
            ket, bra, messages, patch, support, cluster_support, plan
        )
        if operators is None:
            return raw
        if operators.ndim == 3:
            return torch.einsum("vab,zba->zv", operators, raw)
        if operators.ndim == 4:
            return torch.einsum("zvab,zba->zv", operators, raw)
        raise ValueError("matrix-free operators must have shape [V,d,d] or [B,V,d,d]")
    joint_boundary = None
    if plan["readout_boundary_mode"] == "joint_halo":
        joint_boundary = _joint_boundary_messages(
            ket, bra, messages, patch, cluster_support, plan
        )
    return _cluster_contract(
        ket,
        bra,
        messages,
        patch,
        plan,
        support=support,
        joint_boundary=joint_boundary,
        operators=operators,
    )


def _batched_2x2_raw_transitions(
    ket: dict[int, torch.Tensor],
    bra: dict[int, torch.Tensor],
    messages: dict[tuple[int, str], torch.Tensor],
    cache_keys: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
    plan: dict[str, Any],
    *,
    operators_by_cache_key: dict[tuple[tuple[int, ...], tuple[int, ...]], torch.Tensor]
    | None = None,
) -> dict[tuple[tuple[int, ...], tuple[int, ...]], torch.Tensor]:
    """Contract compatible 2x2 readout patches over a patch batch axis."""
    if plan.get("readout_contraction_mode", "einsum") == "matrix_free":
        B = int(next(iter(ket.values())).shape[0])
        result: dict[tuple[tuple[int, ...], tuple[int, ...]], torch.Tensor] = {}

        def contract_requests(requests: list[dict[str, Any]]) -> None:
            if len(requests) == 1:
                request = requests[0]
                request_operands = list(request["operands"])
                request_labels = [list(value) for value in request["labels"]]
                if plan.get("joint_halo_rank") is None:
                    raw = _contract_labeled_pairwise(
                        request_operands,
                        request_labels,
                        list(request["output_labels"]),
                        batch_label=0,
                        memory_limit=int(plan["memory_limit"]),
                        trace_labels=set(request["trace_labels"]),
                        description="matrix-free 2x2 readout contraction",
                    )
                else:
                    raw, planned_peak, used_fallback = _contract_labeled_2x2_bounded(
                        request_operands,
                        request_labels,
                        list(request["output_labels"]),
                        batch_label=0,
                        memory_limit=int(plan["memory_limit"]),
                        trace_labels=set(request["trace_labels"]),
                        operand_positions=list(request["operand_positions"]),
                        description=("matrix-free 2x2 readout contraction"),
                        audit=plan["bounded_2x2_path_audit"],
                    )
                    audit = plan["bounded_2x2_path_audit"]
                    audit["canonical_tree_contraction_count"] += B
                    audit["max_planned_intermediate_elements"] = max(
                        int(audit["max_planned_intermediate_elements"]),
                        int(planned_peak),
                    )
                    if used_fallback:
                        audit["fallback_contraction_count"] += B
                dimension = 2 ** len(request["cache_key"][0])
                raw = raw.reshape(B, dimension, dimension)
                result[request["cache_key"]] = _apply_matrix_free_operators(
                    raw, request["operators"]
                )
                return

            combined_operands = []
            combined_labels = []
            operand_count = len(requests[0]["operands"])
            for operand_index in range(operand_count):
                values = []
                base_labels = list(requests[0]["labels"][operand_index])
                if 0 in base_labels:
                    batch_axis = base_labels.index(0)
                    for request in requests:
                        values.append(request["operands"][operand_index])
                    labels_value = base_labels
                else:
                    batch_axis = 0
                    labels_value = [0, *base_labels]
                    for request in requests:
                        value = request["operands"][operand_index]
                        values.append(value.unsqueeze(0).expand(B, *value.shape))
                combined_operands.append(torch.cat(values, dim=batch_axis))
                combined_labels.append(labels_value)
            try:
                if plan.get("joint_halo_rank") is None:
                    raw = _contract_labeled_pairwise(
                        combined_operands,
                        combined_labels,
                        list(requests[0]["output_labels"]),
                        batch_label=0,
                        memory_limit=int(plan["memory_limit"]),
                        trace_labels=set(requests[0]["trace_labels"]),
                        description=("batched matrix-free 2x2 readout contraction"),
                    )
                else:
                    raw, planned_peak, used_fallback = _contract_labeled_2x2_bounded(
                        combined_operands,
                        combined_labels,
                        list(requests[0]["output_labels"]),
                        batch_label=0,
                        memory_limit=int(plan["memory_limit"]),
                        trace_labels=set(requests[0]["trace_labels"]),
                        operand_positions=list(requests[0]["operand_positions"]),
                        description=("batched matrix-free 2x2 readout contraction"),
                        audit=plan["bounded_2x2_path_audit"],
                    )
                    audit = plan["bounded_2x2_path_audit"]
                    contraction_rows = len(requests) * B
                    audit["canonical_tree_contraction_count"] += contraction_rows
                    audit["max_planned_intermediate_elements"] = max(
                        int(audit["max_planned_intermediate_elements"]),
                        int(planned_peak),
                    )
                    if used_fallback:
                        audit["fallback_contraction_count"] += contraction_rows
            except RuntimeError as error:
                if not plan.get(
                    "auto_readout_tiling", False
                ) or not _is_readout_path_memory_error(error):
                    raise
                middle = len(requests) // 2
                contract_requests(requests[:middle])
                contract_requests(requests[middle:])
                return
            dimension = 2 ** len(requests[0]["cache_key"][0])
            raw = raw.reshape(len(requests), B, dimension, dimension)
            for index, request in enumerate(requests):
                result[request["cache_key"]] = _apply_matrix_free_operators(
                    raw[index], request["operators"]
                )

        # Construct only a bounded number of patch networks at once.  The
        # previous implementation first materialized low-rank halo factors for
        # every target in the layer and applied ``patch_step`` only to the
        # subsequent contraction.  On a large lattice that made local readout
        # storage scale with the number of target patches.  Keep at most
        # ``patch_step`` prepared requests live, while still batching compatible
        # signatures whenever they arrive in the same bounded window.
        patch_step = int(plan["readout_patch_batch_size"])
        pending: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        pending_count = 0

        def flush_largest_pending() -> None:
            nonlocal pending_count
            if not pending:
                return
            signature = max(pending, key=lambda key: len(pending[key]))
            requests = pending.pop(signature)
            pending_count -= len(requests)
            contract_requests(requests)

        for cache_key in cache_keys:
            support, readout_support = cache_key
            patch = _select_readout_patch(readout_support, plan)
            if len(patch) != 4:
                raise ValueError("matrix-free patch batching requires 2x2 patches")
            (
                operands,
                labels,
                output_labels,
                trace_labels,
                network_B,
                operand_positions,
            ) = _matrix_free_2x2_network(
                ket, bra, messages, patch, support, readout_support, plan
            )
            if network_B != B:
                raise RuntimeError("matrix-free network batch mismatch")
            min_row = min(value[0] for value in operand_positions)
            min_col = min(value[1] for value in operand_positions)
            relative_positions = [
                (row - min_row, col - min_col) for row, col in operand_positions
            ]
            operand_signature = []
            for operand, operand_labels in zip(operands, labels):
                shape = list(int(x) for x in operand.shape)
                if 0 in operand_labels:
                    del shape[operand_labels.index(0)]
                operand_signature.append(
                    (tuple(operand_labels), 0 in operand_labels, tuple(shape))
                )
            signature = (
                tuple(operand_signature),
                tuple(output_labels),
                tuple(sorted(trace_labels)),
                len(support),
                tuple(relative_positions),
            )
            pending.setdefault(signature, []).append(
                {
                    "cache_key": cache_key,
                    "operands": operands,
                    "labels": labels,
                    "output_labels": output_labels,
                    "trace_labels": trace_labels,
                    "operand_positions": relative_positions,
                    "operators": (
                        None
                        if operators_by_cache_key is None
                        else operators_by_cache_key[cache_key]
                    ),
                }
            )
            pending_count += 1
            if len(pending[signature]) >= patch_step:
                requests = pending.pop(signature)
                pending_count -= len(requests)
                contract_requests(requests)
                del requests
            elif pending_count >= patch_step:
                flush_largest_pending()
            # Do not let loop locals retain the just-contracted network while
            # the next patch is being constructed.
            del operands, labels, output_labels, trace_labels, operand_positions
        while pending:
            flush_largest_pending()
        return result
    B = int(next(iter(ket.values())).shape[0])
    direction_order = {direction: index for index, direction in enumerate("UDLR")}
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    fallback = []
    for cache_key in cache_keys:
        support, readout_support = cache_key
        patch = _select_readout_patch(readout_support, plan)
        if len(patch) != 4:
            fallback.append(cache_key)
            continue
        local_by_site = {site: local for local, site in enumerate(patch)}
        joint_boundary = None
        if plan["readout_boundary_mode"] == "joint_halo":
            joint_boundary = _joint_boundary_messages(
                ket, bra, messages, patch, readout_support, plan
            )
        joint_boundary = joint_boundary or {}
        joint_edges = {
            (site, direction)
            for direction, (sites, _) in joint_boundary.items()
            for site in sites
        }
        external = []
        ordinary_messages = {}
        for site in patch:
            local = local_by_site[site]
            for direction in "UDLR":
                other = plan["neighbors"][site].get(direction)
                if other is None or other in local_by_site:
                    continue
                is_joint = (site, direction) in joint_edges
                external.append((local, direction, is_joint))
                if not is_joint:
                    ordinary_messages[(local, direction)] = messages[
                        (other, OPPOSITE[direction])
                    ]
        external.sort(key=lambda item: (item[0], direction_order[item[1]]))
        canonical_joint = {}
        for direction, (sites, value) in joint_boundary.items():
            canonical_joint[direction] = (
                tuple(local_by_site[site] for site in sites),
                value,
            )
        canonical_support = tuple(local_by_site[site] for site in support)
        signature = (
            canonical_support,
            (
                None
                if operators_by_cache_key is None
                else tuple(int(x) for x in operators_by_cache_key[cache_key].shape)
            ),
            tuple(tuple(int(x) for x in ket[site].shape[1:]) for site in patch),
            tuple(tuple(int(x) for x in bra[site].shape) for site in patch),
            tuple(
                (
                    local,
                    direction,
                    is_joint,
                    None
                    if is_joint
                    else tuple(
                        int(x) for x in ordinary_messages[(local, direction)].shape[1:]
                    ),
                )
                for local, direction, is_joint in external
            ),
            tuple(
                (direction, sites, tuple(int(x) for x in value.shape[1:]))
                for direction, (sites, value) in sorted(canonical_joint.items())
            ),
        )
        groups.setdefault(signature, []).append(
            {
                "cache_key": cache_key,
                "patch": patch,
                "support": canonical_support,
                "external": tuple(external),
                "ordinary_messages": ordinary_messages,
                "joint": canonical_joint,
                "operators": (
                    None
                    if operators_by_cache_key is None
                    else operators_by_cache_key[cache_key]
                ),
            }
        )

    result = {}
    patch_step = int(plan["readout_patch_batch_size"])
    for requests in groups.values():
        for start in range(0, len(requests), patch_step):
            selected = requests[start : start + patch_step]
            if len(selected) == 1:
                request = selected[0]
                support, readout_support = request["cache_key"]
                result[request["cache_key"]] = _raw_transition(
                    ket,
                    bra,
                    messages,
                    support,
                    plan,
                    readout_support=readout_support,
                    operators=request["operators"],
                )
                continue

            synthetic_ket = {}
            synthetic_bra = {}
            for local in range(4):
                synthetic_ket[local] = torch.cat(
                    [ket[request["patch"][local]] for request in selected], dim=0
                )
                bra_parts = []
                for request in selected:
                    site = request["patch"][local]
                    value = bra[site]
                    if value.ndim == ket[site].ndim:
                        if int(value.shape[0]) != B:
                            raise ValueError("batched readout bra has wrong batch size")
                        bra_parts.append(value)
                    else:
                        bra_parts.append(value.unsqueeze(0).expand(B, *value.shape))
                synthetic_bra[local] = torch.cat(bra_parts, dim=0)

            synthetic_neighbors = {
                0: {"R": 1, "D": 2},
                1: {"L": 0, "D": 3},
                2: {"U": 0, "R": 3},
                3: {"U": 1, "L": 2},
            }
            synthetic_messages = {}
            ghost = 4
            for local, direction, is_joint in selected[0]["external"]:
                synthetic_neighbors[local][direction] = ghost
                if not is_joint:
                    synthetic_messages[(ghost, OPPOSITE[direction])] = torch.cat(
                        [
                            request["ordinary_messages"][(local, direction)]
                            for request in selected
                        ],
                        dim=0,
                    )
                ghost += 1
            synthetic_joint = {}
            for direction, (sites, _) in selected[0]["joint"].items():
                synthetic_joint[direction] = (
                    sites,
                    torch.cat(
                        [request["joint"][direction][1] for request in selected], dim=0
                    ),
                )
            synthetic_plan = {
                "neighbors": synthetic_neighbors,
                "memory_limit": plan["memory_limit"],
            }
            synthetic_operators = None
            if selected[0]["operators"] is not None:
                synthetic_operators = torch.cat(
                    [
                        request["operators"]
                        .unsqueeze(0)
                        .expand(B, *request["operators"].shape)
                        for request in selected
                    ],
                    dim=0,
                )
            try:
                raw = _cluster_contract(
                    synthetic_ket,
                    synthetic_bra,
                    synthetic_messages,
                    (0, 1, 2, 3),
                    synthetic_plan,
                    support=selected[0]["support"],
                    joint_boundary=synthetic_joint,
                    operators=synthetic_operators,
                )
            except RuntimeError as error:
                if (
                    not plan.get("auto_readout_tiling", False)
                    or len(selected) == 1
                    or not _is_readout_path_memory_error(error)
                ):
                    raise
                for request in selected:
                    request_key = request["cache_key"]
                    request_operators = (
                        None
                        if operators_by_cache_key is None
                        else {request_key: operators_by_cache_key[request_key]}
                    )
                    result.update(
                        _batched_2x2_raw_transitions(
                            ket,
                            bra,
                            messages,
                            (request_key,),
                            plan,
                            operators_by_cache_key=request_operators,
                        )
                    )
                continue
            if synthetic_operators is None:
                dimension = 2 ** len(selected[0]["support"])
                raw = raw.reshape(len(selected), B, dimension, dimension)
            else:
                raw = raw.reshape(len(selected), B, int(synthetic_operators.shape[1]))
            for local, request in enumerate(selected):
                result[request["cache_key"]] = raw[local]

    for cache_key in fallback:
        support, readout_support = cache_key
        result[cache_key] = _raw_transition(
            ket,
            bra,
            messages,
            support,
            plan,
            readout_support=readout_support,
            operators=(
                None
                if operators_by_cache_key is None
                else operators_by_cache_key[cache_key]
            ),
        )
    return result


def _readout_region(
    support: tuple[int, ...],
    plan: dict[str, Any],
    *,
    readout_support: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    """Return the sites represented by the standard Bethe region factor."""
    cluster_support = readout_support or support
    patch = (
        tuple(support)
        if plan["readout_block"] == "support"
        else _select_readout_patch(cluster_support, plan)
    )
    region = set(patch)
    if (
        plan["readout_block"] != "support"
        and plan["readout_boundary_mode"] == "joint_halo"
    ):
        for direction in "UDLR":
            side_sites_all = _patch_side_sites(patch, direction, plan)
            if len(side_sites_all) < 2:
                # A one-leg end uses its ordinary incoming BP message; it is
                # not represented explicitly by a correlated strip factor.
                continue
            side_sites = _joint_side_patch_sites(
                patch, cluster_support, direction, plan
            )
            outside = tuple(
                plan["neighbors"][site].get(direction) for site in side_sites
            )
            if all(site is not None for site in outside):
                region.update(int(site) for site in outside)
    return tuple(sorted(region))


def _region_factor(
    support: tuple[int, ...],
    factors: dict[str, Any],
    plan: dict[str, Any],
    *,
    readout_support: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Return prod(region site factors) / prod(region internal edges)."""
    region = set(_readout_region(support, plan, readout_support=readout_support))
    internal = {
        edge for edge in factors["edge"] if edge[0] in region and edge[1] in region
    }
    ref = next(iter(factors["site"].values()))
    value = torch.ones_like(ref)
    for site, factor in factors["site"].items():
        if site in region:
            value *= factor
    for edge, factor in factors["edge"].items():
        if edge in internal:
            value /= factor
    return value


def _outside_factor(
    support: tuple[int, ...],
    factors: dict[str, Any],
    plan: dict[str, Any],
    *,
    readout_support: tuple[int, ...] | None = None,
) -> torch.Tensor:
    region = set(_readout_region(support, plan, readout_support=readout_support))
    internal = {
        edge for edge in factors["edge"] if edge[0] in region and edge[1] in region
    }
    ref = next(iter(factors["site"].values()))
    value = torch.ones_like(ref)
    for site, factor in factors["site"].items():
        if site not in region:
            value *= factor
    for edge, factor in factors["edge"].items():
        if edge not in internal:
            value /= factor
    return value


def _outside_factor_audit(
    support: tuple[int, ...],
    factors: dict[str, Any],
    plan: dict[str, Any],
    *,
    readout_support: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Describe the Bethe region used by ``_outside_factor``.

    In ``single_site`` boundary mode the region is exactly the patch and every
    crossing edge remains in the outside denominator.  In the historical
    ``joint_halo`` mode, an adjacent halo pair can also be represented by the
    raw readout.  Keeping this audit next to the factor implementation makes it
    possible to distinguish a region-bookkeeping mismatch from a bad
    transition matrix in the benchmark ledger.
    """
    cluster_support = readout_support or support
    patch = (
        tuple(support)
        if plan["readout_block"] == "support"
        else _select_readout_patch(cluster_support, plan)
    )
    region = set(_readout_region(support, plan, readout_support=readout_support))
    halo_by_side: dict[str, list[int]] = {}
    separator_by_side: dict[str, list[list[int]]] = {}
    if (
        plan["readout_block"] != "support"
        and plan["readout_boundary_mode"] == "joint_halo"
    ):
        for direction in "UDLR":
            side_sites_all = _patch_side_sites(patch, direction, plan)
            if len(side_sites_all) < 2:
                continue
            side_sites = _joint_side_patch_sites(
                patch, cluster_support, direction, plan
            )
            outside = tuple(
                plan["neighbors"][site].get(direction) for site in side_sites
            )
            if all(site is not None for site in outside):
                halo = sorted({int(site) for site in outside})
                region.update(halo)
                halo_by_side[direction] = halo
            else:
                separator_by_side[direction] = [
                    [int(site), int(outside_site)]
                    for site, outside_site in zip(side_sites, outside)
                    if outside_site is not None
                ]
    internal = sorted(
        [
            list(edge)
            for edge in factors["edge"]
            if edge[0] in region and edge[1] in region
        ]
    )
    crossing = sorted(
        [
            list(edge)
            for edge in factors["edge"]
            if (edge[0] in region) != (edge[1] in region)
        ]
    )
    outside_sites = sorted(int(site) for site in factors["site"] if site not in region)
    outside_edges = sorted(
        [list(edge) for edge in factors["edge"] if list(edge) not in internal]
    )
    return {
        "support": [int(site) for site in support],
        "readout_support": [int(site) for site in cluster_support],
        "patch": [int(site) for site in patch],
        "region": sorted(int(site) for site in region),
        "boundary_mode": plan["readout_boundary_mode"],
        "halo_by_side": halo_by_side,
        "separator_edges_by_side": separator_by_side,
        "internal_edges": internal,
        "crossing_edges": crossing,
        "outside_site_factor_sites": outside_sites,
        "outside_edge_denominator_edges": outside_edges,
        "site_partition": len(outside_sites) + len(region),
        "edge_partition": len(factors["edge"]),
        "internal_edge_count": len(internal),
        "region_decomposition_closed": (
            set(region).issubset(set(factors["site"]))
            and len(outside_sites) + len(region) == len(factors["site"])
        ),
    }


def _read_target_chunk(
    ket: dict[int, torch.Tensor],
    clean: dict[str, Any],
    messages: dict[tuple[int, str], torch.Tensor],
    target: dict[str, Any],
    plan: dict[str, Any],
    *,
    source_amplitudes: torch.Tensor,
    support_diagnostics: list[dict[str, Any]] | None = None,
    branch_ids: torch.Tensor | None = None,
    source_gate_ids: torch.Tensor | None = None,
    source_mode_ids: torch.Tensor | None = None,
    bp_residuals: torch.Tensor | None = None,
    target_layer: int | None = None,
    branch_diagnostic_pairs: set[tuple[int, int]] | None = None,
    branch_diagnostics: list[dict[str, Any]] | None = None,
    exact_source_amplitudes: dict[int, torch.Tensor] | None = None,
    quarantine_ledger: list[dict[str, Any]] | None = None,
    kappa_warning_ledger: list[dict[str, Any]] | None = None,
    fallback_queue: list[dict[str, Any]] | None = None,
    readout_modes: tuple[str, ...] | None = None,
    bethe_factors_override: dict[str, Any] | None = None,
    gate_ids_override: tuple[int, ...] | None = None,
    one_qubit_reuse_plan: dict[str, Any] | None = None,
) -> Any:
    B = int(next(iter(ket.values())).shape[0])
    source_amplitudes = source_amplitudes.reshape(-1)
    if int(source_amplitudes.numel()) != B:
        raise ValueError(
            "source amplitude batch does not match mixed-ket batch: "
            f"{int(source_amplitudes.numel())} != {B}"
        )
    if bool((~torch.isfinite(source_amplitudes)).any()):
        raise RuntimeError("non-finite source-layer amplitudes")
    diagnostic_enabled = (
        branch_diagnostic_pairs is not None
        and branch_diagnostics is not None
        and branch_ids is not None
        and source_gate_ids is not None
        and source_mode_ids is not None
        and bp_residuals is not None
        and target_layer is not None
    )
    ledger_enabled = (
        (
            quarantine_ledger is not None
            or kappa_warning_ledger is not None
            or fallback_queue is not None
        )
        and branch_ids is not None
        and source_gate_ids is not None
        and source_mode_ids is not None
        and bp_residuals is not None
        and target_layer is not None
    )
    if diagnostic_enabled or ledger_enabled:
        metadata = (branch_ids, source_gate_ids, source_mode_ids, bp_residuals)
        if any(int(value.numel()) != B for value in metadata):
            raise ValueError("branch diagnostic metadata does not match readout batch")
    modes_requested = tuple(readout_modes or (clean["config"].branch_weight_mode,))
    allowed_modes = {
        "source_anchored",
        "source_anchored_sumratio",
        "bethe_ratio",
        "bethe_unnormalized",
    }
    if not modes_requested or any(
        mode not in allowed_modes for mode in modes_requested
    ):
        raise ValueError(f"unsupported readout_modes: {modes_requested}")
    if len(set(modes_requested)) != len(modes_requested):
        raise ValueError("readout_modes must not contain duplicates")
    gate_ids = tuple(
        target["gate_ids"] if gate_ids_override is None else gate_ids_override
    )
    unknown_gate_ids = set(gate_ids) - set(target["gate_ids"])
    if unknown_gate_ids:
        raise ValueError(
            f"gate_ids_override contains unknown targets: {sorted(unknown_gate_ids)}"
        )
    if one_qubit_reuse_plan is not None and modes_requested != ("source_anchored",):
        raise ValueError("1Q readout reuse is restricted to source_anchored readout")
    multi_mode = len(modes_requested) > 1
    device = next(iter(ket.values())).device
    numerator_by_mode = {
        mode: torch.zeros(
            B, len(target["gate_ids"]), dtype=torch.float64, device=device
        )
        for mode in modes_requested
    }
    denominator_by_mode = {
        mode: torch.zeros_like(numerator_by_mode[mode]) for mode in modes_requested
    }
    quarantine_by_mode = {
        mode: torch.zeros(B, len(target["gate_ids"]), dtype=torch.bool, device=device)
        for mode in modes_requested
    }
    raw_cache = {}
    fused_local_cache = {}
    outside_cache = {}
    outside_audit_cache = {}
    closure_cache = {}
    support_q_cache = {}
    conditioning_cache = {}
    target_gate_indices_by_cache_key = {}
    for gate_id in gate_ids:
        target_local = int(target["gate_to_local"][gate_id])
        support = target["supports"][gate_id]
        readout_support = target["readout_supports"][gate_id]
        cache_key = (support, readout_support)
        target_gate_indices_by_cache_key.setdefault(cache_key, []).append(
            (target_local, int(gate_id))
        )
    fused_readout = (
        bool(plan.get("fuse_readout_operators", False))
        and plan.get("local_readout_method", "cluster") == "cluster"
        and plan["readout_block"] == "2x2"
        and not diagnostic_enabled
        and not ledger_enabled
        and support_diagnostics is None
        and not clean["config"].auto_quarantine_ill_conditioned
        and all(
            len(entries) == 1 for entries in target_gate_indices_by_cache_key.values()
        )
    )
    operators_by_cache_key = None
    current_operator_slices = {}
    reuse_operator_slices = {}
    if fused_readout:
        operators_by_cache_key = {}
        for cache_key, entries in target_gate_indices_by_cache_key.items():
            gate_id = entries[0][1]
            support = target["supports"][gate_id]
            dimension = 2 ** len(support)
            gate_data = clean["gate_data"][gate_id]
            modes = gate_data["E_modes"].reshape(-1, dimension, dimension)
            if clean["config"].target_amplitude_filter_tol > 0.0:
                modes = modes[gate_data["target_mode_keep"]]
            identity = torch.eye(dimension, dtype=modes.dtype, device=modes.device)
            operator_parts = [identity.unsqueeze(0), modes]
            current_operator_slices[cache_key] = (1, 1 + int(modes.shape[0]))
            offset = 1 + int(modes.shape[0])
            for reuse_entry in (
                ()
                if one_qubit_reuse_plan is None
                else one_qubit_reuse_plan["by_cache_key"].get(cache_key, ())
            ):
                reuse_modes = reuse_entry["operators"]
                stop = offset + int(reuse_modes.shape[0])
                reuse_operator_slices[int(reuse_entry["gate_id"])] = (offset, stop)
                operator_parts.append(reuse_modes)
                offset = stop
            operators_by_cache_key[cache_key] = torch.cat(operator_parts, dim=0)
    prebatched_raw_cache = {}
    if (
        plan.get("batch_readout_patches", False)
        and plan.get("local_readout_method", "cluster") == "cluster"
        and plan["readout_block"] == "2x2"
    ):
        prebatched_raw_cache = _batched_2x2_raw_transitions(
            ket,
            clean["peps"],
            messages,
            tuple(target_gate_indices_by_cache_key),
            plan,
            operators_by_cache_key=operators_by_cache_key,
        )
    weight_mode = clean["config"].branch_weight_mode
    factors_required = (
        "bethe_ratio" in modes_requested
        or "bethe_unnormalized" in modes_requested
        or support_diagnostics is not None
        or diagnostic_enabled
    )
    factors = bethe_factors_override
    if factors is not None:
        factor_batch = int(factors["Z"].shape[0])
        if factor_batch != B:
            raise ValueError(
                "precomputed Bethe batch does not match readout batch: "
                f"{factor_batch} != {B}"
            )
    if factors_required and factors is None:
        factors = _bethe_factors(ket, clean["peps"], messages, plan)
    clean_z = clean["Z_clean"]
    if (not torch.isfinite(clean_z).item()) or abs(clean_z.item()) <= 1e-30:
        raise RuntimeError("clean Bethe Z is non-finite or zero")
    global_bethe_overlap = None
    if factors is not None:
        global_bethe_overlap = factors["Z"] / clean_z
        if bool((~torch.isfinite(global_bethe_overlap)).any()):
            raise RuntimeError("non-finite global Bethe overlap")
    source_weight = source_amplitudes.abs().square().to(dtype=torch.float64)
    branch_weights = {
        "source_anchored": source_weight,
        "bethe_ratio": (
            global_bethe_overlap.abs().square().to(dtype=torch.float64)
            if global_bethe_overlap is not None
            else None
        ),
    }
    one_qubit_local_cache = {}
    for gate_id in gate_ids:
        support = target["supports"][gate_id]
        readout_support = target["readout_supports"][gate_id]
        cache_key = (support, readout_support)
        if cache_key not in raw_cache:
            if cache_key in prebatched_raw_cache:
                transition = prebatched_raw_cache[cache_key]
            else:
                transition = _raw_transition(
                    ket,
                    clean["peps"],
                    messages,
                    support,
                    plan,
                    readout_support=readout_support,
                    operators=(
                        None
                        if operators_by_cache_key is None
                        else operators_by_cache_key[cache_key]
                    ),
                )
            if fused_readout:
                trace = transition[:, 0]
                current_start, current_stop = current_operator_slices[cache_key]
                fused_local_cache[cache_key] = transition[:, current_start:current_stop]
                if one_qubit_reuse_plan is not None:
                    for reuse_entry in one_qubit_reuse_plan["by_cache_key"].get(
                        cache_key, ()
                    ):
                        reuse_gate_id = int(reuse_entry["gate_id"])
                        reuse_start, reuse_stop = reuse_operator_slices[reuse_gate_id]
                        one_qubit_local_cache[reuse_gate_id] = transition[
                            :, reuse_start:reuse_stop
                        ]
                raw_fro = torch.zeros_like(trace.real)
            else:
                raw_cache[cache_key] = transition
                raw = transition
                trace = torch.diagonal(raw, dim1=1, dim2=2).sum(dim=1)
                raw_fro = torch.linalg.vector_norm(raw.reshape(B, -1), dim=1)
            trace_abs = trace.abs()
            if fused_readout:
                kappa = torch.zeros_like(raw_fro)
                ill_conditioned = (~torch.isfinite(trace)) | (
                    trace_abs <= clean["config"].trace_floor
                )
                warning_condition = torch.zeros_like(ill_conditioned)
            else:
                kappa = torch.where(
                    trace_abs > 0.0,
                    raw_fro / trace_abs,
                    torch.full_like(raw_fro, float("inf")),
                )
                ill_conditioned = (
                    (~torch.isfinite(kappa))
                    | (kappa >= clean["config"].conditional_kappa_skip)
                    | (trace_abs <= clean["config"].trace_floor)
                )
                warning_condition = (
                    kappa >= clean["config"].conditional_kappa_warn
                ) & ~ill_conditioned
            conditioning_cache[cache_key] = (
                trace,
                raw_fro,
                kappa,
                ill_conditioned,
                warning_condition,
            )
            if clean["config"].auto_quarantine_ill_conditioned:
                for target_local, _ in target_gate_indices_by_cache_key[cache_key]:
                    for mode in ("source_anchored", "bethe_ratio"):
                        if mode in quarantine_by_mode:
                            quarantine_by_mode[mode][:, target_local] |= ill_conditioned
            if ledger_enabled:
                target_gate_ids_for_support = [
                    gate_value
                    for _, gate_value in target_gate_indices_by_cache_key[cache_key]
                ]
                source_gate_values = source_gate_ids.detach().cpu().tolist()
                source_mode_values = source_mode_ids.detach().cpu().tolist()
                branch_values = branch_ids.detach().cpu().tolist()
                residual_values = bp_residuals.detach().cpu().tolist()
                source_amplitude_values = (
                    source_amplitudes.abs().detach().cpu().tolist()
                )
                trace_values = trace.detach().cpu().tolist()
                raw_fro_values = raw_fro.detach().cpu().tolist()
                kappa_values = kappa.detach().cpu().tolist()
                ill_values = ill_conditioned.detach().cpu().tolist()
                warning_values = warning_condition.detach().cpu().tolist()
                for position in range(B):
                    common = {
                        "branch_id": int(branch_values[position]),
                        "target_layer": int(target_layer),
                        "source_gate": int(source_gate_values[position]),
                        "source_mode": int(source_mode_values[position]),
                        "target_support": [int(site) for site in support],
                        "target_gate_ids": target_gate_ids_for_support,
                        "trace_real": float(trace_values[position].real),
                        "trace_imag": float(trace_values[position].imag),
                        "trace_abs": float(abs(trace_values[position])),
                        "raw_fro_norm": float(raw_fro_values[position]),
                        "kappa": float(kappa_values[position]),
                        "bp_residual": float(residual_values[position]),
                        "source_amplitude_abs": float(
                            source_amplitude_values[position]
                        ),
                    }
                    if bool(ill_values[position]):
                        entry = {
                            **common,
                            "reason": "ill_conditioned_conditional_ratio",
                            "threshold": float(clean["config"].conditional_kappa_skip),
                        }
                        if (
                            clean["config"].auto_quarantine_ill_conditioned
                            and quarantine_ledger is not None
                        ):
                            quarantine_ledger.append(entry)
                        if (
                            clean["config"].auto_quarantine_ill_conditioned
                            and fallback_queue is not None
                        ):
                            fallback_queue.append(
                                {
                                    **entry,
                                    "resolution": "pending",
                                }
                            )
                        elif kappa_warning_ledger is not None:
                            kappa_warning_ledger.append(
                                {
                                    **entry,
                                    "reason": "ill_conditioned_not_quarantined",
                                }
                            )
                    elif (
                        bool(warning_values[position])
                        and kappa_warning_ledger is not None
                    ):
                        kappa_warning_ledger.append(
                            {
                                **common,
                                "reason": "high_condition_number",
                                "threshold": float(
                                    clean["config"].conditional_kappa_warn
                                ),
                            }
                        )
            if (
                "bethe_unnormalized" in modes_requested
                or support_diagnostics is not None
                or diagnostic_enabled
            ):
                if factors is None:
                    raise RuntimeError("Bethe factors missing for outside factor")
                outside_cache[cache_key] = _outside_factor(
                    support, factors, plan, readout_support=readout_support
                )
                outside_audit_cache[cache_key] = _outside_factor_audit(
                    support, factors, plan, readout_support=readout_support
                )
                if bool((~torch.isfinite(outside_cache[cache_key])).any()):
                    raise RuntimeError("non-finite patch-outside Bethe factor")
                region_factor = _region_factor(
                    support, factors, plan, readout_support=readout_support
                )
                if bool((~torch.isfinite(region_factor)).any()):
                    raise RuntimeError("non-finite Bethe region factor")
                global_z = factors["Z"]
                factor_reconstruction = outside_cache[cache_key] * region_factor
                identity_reconstruction = outside_cache[cache_key] * trace
                scale = global_z.abs().clamp_min(torch.finfo(torch.float64).tiny)
                factor_error = (factor_reconstruction - global_z).abs() / scale
                identity_error = (identity_reconstruction - global_z).abs() / scale
                region_identity_error = (
                    trace - region_factor
                ).abs() / region_factor.abs().clamp_min(torch.finfo(torch.float64).tiny)
                closure_cache[cache_key] = {
                    "region_factor": region_factor,
                    "factor_reconstruction": factor_reconstruction,
                    "identity_reconstruction": identity_reconstruction,
                    "factor_error": factor_error,
                    "identity_error": identity_error,
                    "region_identity_error": region_identity_error,
                }
                strict_tol = clean["config"].strict_region_closure_tol
                if strict_tol is not None:
                    max_factor_error = float(factor_error.max().item())
                    max_identity_error = float(identity_error.max().item())
                    if max_factor_error > strict_tol or max_identity_error > strict_tol:
                        worst = int(identity_error.argmax().item())
                        source_tag = ""
                        if (
                            source_gate_ids is not None
                            and source_mode_ids is not None
                            and bp_residuals is not None
                        ):
                            source_tag = (
                                f", source_gate={int(source_gate_ids[worst].item())}"
                                f", source_mode={int(source_mode_ids[worst].item())}"
                                f", bp_residual="
                                f"{float(bp_residuals[worst].item()):.3e}"
                            )
                        raise RuntimeError(
                            "Bethe region closure failed before target insertion: "
                            f"support={support}, readout_support={readout_support}, "
                            f"boundary={plan['readout_boundary_mode']}, "
                            f"factor_error={max_factor_error:.3e}, "
                            f"identity_error={max_identity_error:.3e}, "
                            f"tolerance={strict_tol:.3e}, "
                            f"worst_branch={worst}{source_tag}, "
                            f"abs_global_Z={float(global_z[worst].abs().item()):.3e}, "
                            f"abs_region_factor="
                            f"{float(region_factor[worst].abs().item()):.3e}, "
                            f"abs_identity_trace={float(trace[worst].abs().item()):.3e}, "
                            f"region_identity_error="
                            f"{float(region_identity_error[worst].item()):.3e}"
                        )
        raw = raw_cache.get(cache_key)
        (
            trace,
            raw_fro,
            kappa,
            ill_conditioned,
            warning_condition,
        ) = conditioning_cache[cache_key]
        if support_diagnostics is not None:
            support_q_cache[cache_key] = (
                outside_cache[cache_key] * trace / clean["Z_clean"]
            )
        dimension = 2 ** len(support)
        gate_data = clean["gate_data"][gate_id]
        modes = gate_data["E_modes"].reshape(-1, dimension, dimension)
        if clean["config"].target_amplitude_filter_tol > 0.0:
            modes = modes[gate_data["target_mode_keep"]]
        local = (
            fused_local_cache[cache_key]
            if fused_readout
            else torch.einsum("vab,zba->zv", modes, raw)
        )
        if bool((~torch.isfinite(local)).any()):
            raise RuntimeError(f"non-finite local target contraction for {support}")
        conditional_skip = (
            ill_conditioned
            if clean["config"].auto_quarantine_ill_conditioned
            else torch.zeros_like(ill_conditioned)
        )
        invalid_trace = (~torch.isfinite(trace)) | (
            trace.abs() <= clean["config"].trace_floor
        )
        conditional_by_mode: dict[str, torch.Tensor] = {}
        good_by_mode: dict[str, torch.Tensor] = {}
        for mode in modes_requested:
            if mode not in ("source_anchored", "bethe_ratio"):
                continue
            if bool((invalid_trace & ~conditional_skip).any()):
                raise RuntimeError(
                    f"zero/non-finite mixed local-rho trace for {support}; "
                    f"trace_floor={clean['config'].trace_floor:.3e}"
                )
            conditional = torch.zeros_like(local)
            good = ~conditional_skip
            if bool(good.any()):
                # This is deliberately confined to the two legacy conditional
                # estimators.  The full-Bethe path below never enters here.
                conditional[good] = local[good] / trace[good].unsqueeze(1)
            conditional_by_mode[mode] = conditional
            good_by_mode[mode] = good
        full_amplitude = None
        if "bethe_unnormalized" in modes_requested:
            outside = outside_cache[cache_key]
            if bool((~torch.isfinite(outside)).any()):
                raise RuntimeError("non-finite patch-outside factor")
            # Strictly unnormalized full-Bethe readout:
            # A_uv = (G_{u,bar P}^Bethe / Z_clean^Bethe) z_uv,P.
            full_amplitude = outside.unsqueeze(1) * local / clean_z
            if bool((~torch.isfinite(full_amplitude)).any()):
                raise RuntimeError("non-finite full-Bethe amplitude")

        bethe_equivalence_error = None
        if "bethe_ratio" in modes_requested and "bethe_unnormalized" in modes_requested:
            ratio_amplitude = (
                global_bethe_overlap.unsqueeze(1) * conditional_by_mode["bethe_ratio"]
            )
            equivalence_scale = torch.maximum(
                ratio_amplitude.abs(), full_amplitude.abs()
            ).amax(dim=1)
            equivalence_difference = (
                (ratio_amplitude - full_amplitude).abs().amax(dim=1)
            )
            bethe_equivalence_error = torch.where(
                equivalence_scale > 0.0,
                equivalence_difference / equivalence_scale,
                equivalence_difference,
            )
            strict_tol = clean["config"].strict_region_closure_tol
            if (
                strict_tol is not None
                and float(bethe_equivalence_error.max().item()) > strict_tol
            ):
                raise RuntimeError(
                    "Bethe-ratio/full-Bethe identity failed before Exact: "
                    f"support={support}, readout_support={readout_support}, "
                    f"max_relative_error="
                    f"{float(bethe_equivalence_error.max().item()):.3e}, "
                    f"tolerance={strict_tol:.3e}"
                )

        if diagnostic_enabled:
            diagnostic_source_gate_ids = source_gate_ids.detach().cpu().tolist()
            diagnostic_source_mode_ids = source_mode_ids.detach().cpu().tolist()
            diagnostic_branch_ids = branch_ids.detach().cpu().tolist()
            diagnostic_residuals = bp_residuals.detach().cpu().tolist()
            diagnostic_target = int(gate_id)
            diagnostic_positions = [
                position
                for position, source_gate in enumerate(diagnostic_source_gate_ids)
                if (int(source_gate), diagnostic_target) in branch_diagnostic_pairs
            ]
            if diagnostic_positions:
                trace_cpu = trace.detach().cpu()
                raw_fro_cpu = raw_fro.detach().cpu()
                local_sq = torch.sum(local.abs().square(), dim=1).detach().cpu()
                amplitude_by_mode = {}
                for mode in modes_requested:
                    if mode == "bethe_unnormalized":
                        amplitude_by_mode[mode] = full_amplitude
                    elif mode == "source_anchored_sumratio":
                        amplitude_by_mode[mode] = source_amplitudes.unsqueeze(1) * local
                    else:
                        amplitude_by_mode[mode] = (
                            source_amplitudes.unsqueeze(1)
                            if mode == "source_anchored"
                            else global_bethe_overlap.unsqueeze(1)
                        ) * conditional_by_mode[mode]
                contribution_by_mode = {
                    mode: torch.sum(value.abs().square(), dim=1).detach().cpu()
                    for mode, value in amplitude_by_mode.items()
                }
                exact_source_values = [None] * B
                if exact_source_amplitudes is not None:
                    for position in diagnostic_positions:
                        gate_value = int(diagnostic_source_gate_ids[position])
                        mode_value = int(diagnostic_source_mode_ids[position])
                        exact_value = exact_source_amplitudes[gate_value].reshape(-1)
                        value = exact_value[mode_value].detach().cpu()
                        exact_source_values[position] = {
                            "real": float(value.real.item()),
                            "imag": float(value.imag.item()),
                            "abs": float(value.abs().item()),
                        }
                target_F = (
                    clean["gate_data"][gate_id]["F_target"]
                    if clean["config"].target_amplitude_filter_tol > 0.0
                    else clean["gate_data"][gate_id]["F"]
                )
                target_F_value = float(target_F.detach().cpu().item())
                for position in diagnostic_positions:
                    trace_value = trace_cpu[position]
                    raw_fro_value = float(raw_fro_cpu[position].item())
                    trace_abs = float(trace_value.abs().item())
                    source_value = source_amplitudes[position].detach().cpu()
                    global_value = (
                        global_bethe_overlap[position].detach().cpu()
                        if global_bethe_overlap is not None
                        else None
                    )
                    patch_value = (
                        (outside_cache[cache_key] * trace / clean_z)[position]
                        .detach()
                        .cpu()
                        if cache_key in outside_cache
                        else None
                    )
                    closure_values = closure_cache.get(cache_key)
                    outside_value = (
                        outside_cache[cache_key][position].detach().cpu()
                        if cache_key in outside_cache
                        else None
                    )

                    def _complex_parts(value: Any) -> dict[str, float] | None:
                        if value is None:
                            return None
                        return {
                            "real": float(value.real.item()),
                            "imag": float(value.imag.item()),
                            "abs": float(value.abs().item()),
                        }

                    source_parts = _complex_parts(source_value)
                    global_parts = _complex_parts(global_value)
                    patch_parts = _complex_parts(patch_value)

                    def _relative_complex_error(
                        left: Any,
                        right: Any,
                        floor: float = 1e-30,
                    ) -> float | None:
                        if left is None or right is None:
                            return None
                        return float(
                            abs(left - right).item()
                            / max(float(abs(right).item()), floor)
                        )

                    epsilon_global = _relative_complex_error(global_value, source_value)
                    epsilon_patch = _relative_complex_error(patch_value, source_value)
                    epsilon_decomp = _relative_complex_error(patch_value, global_value)
                    branch_diagnostics.append(
                        {
                            "branch_id": int(diagnostic_branch_ids[position]),
                            "target_layer": int(target_layer),
                            "source_gate": int(diagnostic_source_gate_ids[position]),
                            "source_mode": int(diagnostic_source_mode_ids[position]),
                            "target_support": [int(site) for site in support],
                            "readout_support": [int(site) for site in readout_support],
                            "target_gate_ids": [
                                int(target_gate_value)
                                for _, target_gate_value in target_gate_indices_by_cache_key[
                                    cache_key
                                ]
                            ],
                            "target_gate": diagnostic_target,
                            "target_mode_count": int(modes.shape[0]),
                            "target_F": target_F_value,
                            "bp_converged": True,
                            "bp_residual": float(diagnostic_residuals[position]),
                            "source_amplitude": source_parts,
                            "bp_source_abs": float(source_value.abs().item()),
                            "exact_source_amplitude": exact_source_values[position],
                            "global_bethe_overlap": global_parts,
                            "patch_bethe_overlap": patch_parts,
                            "epsilon_global": epsilon_global,
                            "epsilon_patch": epsilon_patch,
                            "epsilon_decomp": epsilon_decomp,
                            "epsilon_factor_decomp": (
                                float(
                                    closure_values["factor_error"][position]
                                    .detach()
                                    .cpu()
                                    .item()
                                )
                                if closure_values is not None
                                else None
                            ),
                            "epsilon_region_identity": (
                                float(
                                    closure_values["region_identity_error"][position]
                                    .detach()
                                    .cpu()
                                    .item()
                                )
                                if closure_values is not None
                                else None
                            ),
                            "epsilon_bethe_ratio_vs_unnormalized": (
                                float(
                                    bethe_equivalence_error[position]
                                    .detach()
                                    .cpu()
                                    .item()
                                )
                                if bethe_equivalence_error is not None
                                else None
                            ),
                            "outside_factor": _complex_parts(outside_value),
                            "outside_factor_audit": outside_audit_cache.get(cache_key),
                            "clean_bethe_Z": _complex_parts(clean_z.detach().cpu()),
                            "trace_real": float(trace_value.real.item()),
                            "trace_imag": float(trace_value.imag.item()),
                            "trace_abs": trace_abs,
                            "raw_fro": raw_fro_value,
                            "frobenius_norm_raw": raw_fro_value,
                            "kappa": (
                                raw_fro_value / trace_abs
                                if trace_abs > 0.0
                                else float("inf")
                            ),
                            "kappa_raw_over_trace": (
                                raw_fro_value / trace_abs
                                if trace_abs > 0.0
                                else float("inf")
                            ),
                            "trace_floor": float(clean["config"].trace_floor),
                            "ill_conditioned": bool(ill_conditioned[position].item()),
                            "conditional_readout_skipped": bool(
                                conditional_skip[position].item()
                            ),
                            "sum_abs_local_v_sq": float(local_sq[position].item()),
                            "branch_contribution_S_u": {
                                mode: float(values[position].item())
                                for mode, values in contribution_by_mode.items()
                            },
                            "sum_abs_anchored_v_sq": float(
                                contribution_by_mode.get(
                                    "source_anchored", torch.zeros(B)
                                )[position].item()
                            ),
                            "sum_abs_bethe_ratio_v_sq": float(
                                contribution_by_mode.get("bethe_ratio", torch.zeros(B))[
                                    position
                                ].item()
                            ),
                            "sum_abs_bethe_unnormalized_v_sq": float(
                                contribution_by_mode.get(
                                    "bethe_unnormalized", torch.zeros(B)
                                )[position].item()
                            ),
                        }
                    )
        local_index = target["gate_to_local"][gate_id]
        for mode in modes_requested:
            if mode == "source_anchored_sumratio":
                numerator_by_mode[mode][:, local_index] = source_weight * torch.sum(
                    local.abs().square(), dim=1, dtype=torch.float64
                )
                denominator_by_mode[mode][:, local_index] = (
                    source_weight * trace.abs().square().to(torch.float64)
                )
                continue
            if mode == "bethe_unnormalized":
                amplitude_uv = full_amplitude
                good = torch.ones(B, dtype=torch.bool, device=device)
                branch_denominator = source_weight
            else:
                conditional = conditional_by_mode[mode]
                if mode == "source_anchored":
                    branch_amplitude = source_amplitudes
                else:
                    branch_amplitude = global_bethe_overlap
                amplitude_uv = branch_amplitude.unsqueeze(1) * conditional
                good = good_by_mode[mode]
                branch_denominator = branch_weights[mode]
            numerator_by_mode[mode][:, local_index] = torch.sum(
                amplitude_uv.abs().square(), dim=1, dtype=torch.float64
            )
            denominator_by_mode[mode][:, local_index] = branch_denominator * good.to(
                dtype=branch_denominator.dtype
            )
            quarantine_by_mode[mode][:, local_index] |= (
                clean["config"].auto_quarantine_ill_conditioned
                and mode in ("source_anchored", "bethe_ratio")
                and conditional_skip
            )
    if support_diagnostics is not None and len(support_q_cache) >= 2:
        values = torch.stack(tuple(support_q_cache.values()), dim=1)
        center = values.mean(dim=1)
        spread = (values - center[:, None]).abs().amax(dim=1)
        support_diagnostics.append(
            {
                "support_count": int(values.shape[1]),
                "max_abs_spread": float(spread.max().detach().cpu().item()),
                "mean_abs_spread": float(spread.mean().detach().cpu().item()),
            }
        )
    for mode in modes_requested:
        if bool((~torch.isfinite(numerator_by_mode[mode])).any()) or bool(
            (~torch.isfinite(denominator_by_mode[mode])).any()
        ):
            raise RuntimeError(f"non-finite {mode} readout")
    one_qubit_reuse_result = None
    if one_qubit_reuse_plan is not None:
        reuse_num = torch.zeros(
            B,
            int(one_qubit_reuse_plan["target_gate_count"]),
            dtype=torch.float64,
            device=device,
        )
        reuse_den = torch.zeros_like(reuse_num)
        for cache_key, entries in one_qubit_reuse_plan["by_cache_key"].items():
            if cache_key not in conditioning_cache:
                raise RuntimeError("1Q readout reuse parent was not contracted")
            trace = conditioning_cache[cache_key][0]
            invalid_trace = (~torch.isfinite(trace)) | (
                trace.abs() <= clean["config"].trace_floor
            )
            if bool(invalid_trace.any()):
                raise RuntimeError("zero/non-finite mixed trace in 1Q readout reuse")
            for entry in entries:
                gate_id = int(entry["gate_id"])
                local = one_qubit_local_cache[gate_id]
                conditional = local / trace.unsqueeze(1)
                target_local = int(entry["target_local"])
                reuse_num[:, target_local] = source_weight * torch.sum(
                    conditional.abs().square(), dim=1, dtype=torch.float64
                )
                reuse_den[:, target_local] = source_weight
        one_qubit_reuse_result = (reuse_num, reuse_den)
    if readout_modes is None:
        mode = modes_requested[0]
        primary = (
            numerator_by_mode[mode],
            denominator_by_mode[mode],
            quarantine_by_mode[mode],
        )
        if one_qubit_reuse_result is not None:
            return {
                "primary": primary,
                "one_qubit_reuse": one_qubit_reuse_result,
            }
        return primary
    return {
        "by_mode": {
            mode: (
                numerator_by_mode[mode],
                denominator_by_mode[mode],
                quarantine_by_mode[mode],
            )
            for mode in modes_requested
        }
    }


def _concat_readout_results(left: Any, right: Any) -> Any:
    if isinstance(left, tuple):
        return tuple(
            torch.cat((left_value, right_value), dim=0)
            for left_value, right_value in zip(left, right)
        )
    if isinstance(left, dict) and "primary" in left:
        return {
            "primary": _concat_readout_results(left["primary"], right["primary"]),
            "one_qubit_reuse": _concat_readout_results(
                left["one_qubit_reuse"], right["one_qubit_reuse"]
            ),
        }
    if isinstance(left, dict) and "by_mode" in left:
        return {
            "by_mode": {
                mode: tuple(
                    torch.cat((left_value, right_value), dim=0)
                    for left_value, right_value in zip(
                        left["by_mode"][mode], right["by_mode"][mode]
                    )
                )
                for mode in left["by_mode"]
            },
        }
    raise TypeError("unsupported tiled readout result")


def _read_target_chunk_auto_tiled(
    ket: dict[int, torch.Tensor],
    clean: dict[str, Any],
    messages: dict[tuple[int, str], torch.Tensor],
    target: dict[str, Any],
    plan: dict[str, Any],
    **kwargs: Any,
) -> Any:
    """Retry an exact readout on smaller branch tiles when paths exceed budget."""
    tiling_stats = kwargs.pop("tiling_stats", None)
    B = int(next(iter(ket.values())).shape[0])
    _emit_stage_profile_event("readout_auto_tiling", "begin", batch_rows=B)
    if tiling_stats is not None:
        tiling_stats["attempts"] += 1
    try:
        result = _read_target_chunk(ket, clean, messages, target, plan, **kwargs)
        if tiling_stats is not None:
            tiling_stats["successful_branch_tiles"].append(B)
        _emit_stage_profile_event(
            "readout_auto_tiling", "end", batch_rows=B, outcome="success"
        )
        return result
    except RuntimeError as error:
        if (
            not plan.get("auto_readout_tiling", False)
            or B <= 1
            or not _is_readout_path_memory_error(error)
        ):
            raise
        if tiling_stats is not None:
            tiling_stats["branch_splits"] += 1
        _emit_stage_profile_event("readout_auto_tiling", "split", batch_rows=B)
        middle = B // 2

        def sliced_kwargs(start: int, stop: int) -> dict[str, Any]:
            sliced = dict(kwargs)
            for name in (
                "source_amplitudes",
                "branch_ids",
                "source_gate_ids",
                "source_mode_ids",
                "bp_residuals",
            ):
                value = sliced.get(name)
                if value is not None:
                    sliced[name] = value[start:stop]
            factors = sliced.get("bethe_factors_override")
            if factors is not None:
                sliced["bethe_factors_override"] = _slice_bethe_factors(
                    factors, start, stop
                )
            return sliced

        left = _read_target_chunk_auto_tiled(
            {site: value[:middle] for site, value in ket.items()},
            clean,
            {key: value[:middle] for key, value in messages.items()},
            target,
            plan,
            tiling_stats=tiling_stats,
            **sliced_kwargs(0, middle),
        )
        right = _read_target_chunk_auto_tiled(
            {site: value[middle:] for site, value in ket.items()},
            clean,
            {key: value[middle:] for key, value in messages.items()},
            target,
            plan,
            tiling_stats=tiling_stats,
            **sliced_kwargs(middle, B),
        )
        result = _concat_readout_results(left, right)
        _emit_stage_profile_event(
            "readout_auto_tiling", "end", batch_rows=B, outcome="split_success"
        )
        return result


def _store_clean_layer(
    ctx: Any,
    clean_cache: dict[str, Any],
    layer: int,
    peps: dict[int, torch.Tensor],
    messages: dict[tuple[int, str], torch.Tensor],
    z_clean: torch.Tensor,
    bp_info: dict[str, Any],
) -> None:
    """Cache one clean layer from either a B=1 or co-batched BP solve."""
    config = clean_cache["config"]
    plan = clean_cache["plan"]
    dressed = clean_cache["dressed"]
    clean_batch = {site: value.unsqueeze(0) for site, value in peps.items()}
    batched_messages = {key: value.unsqueeze(0) for key, value in messages.items()}
    gate_data = {}
    raw_cache = {}
    required_layers = clean_cache.get("gate_data_layers")
    gates_for_data = (
        ctx.layers[layer] if required_layers is None or layer in required_layers else ()
    )
    for gate in gates_for_data:
        gate_id = int(gate["gate_idx"])
        support = tuple(int(q) for q in gate["qubits"])
        readout_support = plan["targets"][layer]["readout_supports"][gate_id]
        cache_key = (support, readout_support)
        if cache_key not in raw_cache:
            raw_cache[cache_key] = _raw_transition(
                clean_batch,
                peps,
                batched_messages,
                support,
                plan,
                readout_support=readout_support,
            )
        raw = raw_cache[cache_key]
        dimension = 2 ** len(support)
        modes = dressed[layer][gate_id].reshape(-1, dimension, dimension)
        local = torch.einsum("uab,zba->zu", modes, raw)
        trace = torch.diagonal(raw, dim1=1, dim2=2).sum(dim=1)
        if bool(((~torch.isfinite(trace)) | (trace.abs() <= 1e-30)).any()):
            raise RuntimeError(f"zero/non-finite clean local-rho trace for {support}")
        amplitudes = (local / trace.unsqueeze(1))[0]
        if bool((~torch.isfinite(amplitudes)).any()):
            raise RuntimeError(f"non-finite clean amplitudes for gate {gate_id}")
        target_keep = amplitudes.abs() >= config.target_amplitude_filter_tol
        if not bool(target_keep.any()):
            raise RuntimeError(f"target filter removed every mode of gate {gate_id}")
        F_full = torch.sum(amplitudes.abs() ** 2, dtype=torch.float64)
        F_target = torch.sum(amplitudes[target_keep].abs() ** 2, dtype=torch.float64)
        gate_data[gate_id] = {
            "support": support,
            "readout_support": readout_support,
            "E_modes": modes,
            "a_modes": amplitudes,
            "target_mode_keep": target_keep,
            "target_filtered_modes": int((~target_keep).sum().item()),
            "target_discarded_F_weight": float((F_full - F_target).item()),
            "F": F_full,
            "F_target": F_target,
        }

    stored_layer = {
        "peps": dict(peps),
        "messages": messages,
        "Z_clean": z_clean,
        "config": config,
        "gate_data": gate_data,
        "bp_info": bp_info,
    }
    offload_device = clean_cache.get("offload_device")
    if offload_device is not None:
        stored_layer = _move_clean_layer_tensors(
            stored_layer, torch.device(offload_device)
        )
    clean_cache["layers"][layer] = stored_layer

    if next(iter(peps.values())).device.type == "cuda":
        storage_bytes = sum(
            int(value.numel()) * int(value.element_size()) for value in peps.values()
        )
        device = next(iter(peps.values())).device
        print(
            f"[clean-layer] layer={layer} "
            f"peps_storage_gib={storage_bytes / 1024.0**3:.3f} "
            f"cache_device={offload_device or 'cuda'} "
            f"allocated_gib="
            f"{torch.cuda.memory_allocated(device) / 1024.0**3:.3f}",
            flush=True,
        )


def _move_clean_layer_tensors(
    layer: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Move the tensor payload of one clean-cache layer to ``device``."""
    gate_data = {
        int(gate_id): {
            key: (value.to(device=device) if isinstance(value, torch.Tensor) else value)
            for key, value in data.items()
        }
        for gate_id, data in layer["gate_data"].items()
    }
    return {
        "peps": {
            int(site): value.to(device=device) for site, value in layer["peps"].items()
        },
        "messages": {
            key: value.to(device=device) for key, value in layer["messages"].items()
        },
        "Z_clean": layer["Z_clean"].to(device=device),
        "config": layer["config"],
        "gate_data": gate_data,
        "bp_info": layer["bp_info"],
    }


def _extend_clean_cache(
    ctx: Any,
    clean_cache: dict[str, Any],
    through_layer: int,
) -> None:
    """Advance missing clean layers with ordinary B=1 BP."""
    config = clean_cache["config"]
    plan = clean_cache["plan"]
    neighbors = plan["neighbors"]
    layers = clean_cache["layers"]
    background_layer_hook = clean_cache.get("background_layer_hook")
    first = len(layers)
    if first > through_layer:
        return
    if first == 0:
        peps = _zero_peps(ctx)
        previous_messages = None
        previous_z = None
    else:
        previous = _move_clean_layer_tensors(
            layers[first - 1], torch.device(ctx.device)
        )
        peps = dict(previous["peps"])
        previous_messages = previous["messages"]
        previous_z = previous["Z_clean"]

    for layer in range(first, through_layer + 1):
        gates = ctx.layers[layer]
        batched = {site: value.unsqueeze(0) for site, value in peps.items()}
        background_invalidated: set[tuple[int, str]] = set()
        if background_layer_hook is not None:
            returned = background_layer_hook(
                batched,
                layer,
                config=config,
                neighbors=neighbors,
            )
            if returned is not None:
                background_invalidated.update(returned)
        invalidated = _apply_layer(batched, gates, config=config, neighbors=neighbors)
        invalidated.update(background_invalidated)
        peps = {site: value[0] for site, value in batched.items()}
        clean_batch = {site: value.unsqueeze(0) for site, value in peps.items()}

        if (
            previous_messages is not None
            and _is_one_qubit_layer(gates)
            and background_layer_hook is None
        ):
            messages = previous_messages
            z_clean = previous_z
            bp_info = {
                "iterations": 0,
                "iterations_per_branch": [0],
                "residual_per_branch": [0.0],
                "projective_raw_residual_per_branch": [0.0],
                "raw_map_residual_per_branch": [0.0],
                "step_residual_per_branch": [0.0],
                "init": "exact_1q_reuse",
            }
        else:
            initial, init_label = _initialize_messages(
                clean_batch,
                peps,
                plan,
                warm=previous_messages,
                clean=None,
                invalidated=invalidated,
            )
            solved, bp_info = _solve_bp(
                clean_batch,
                peps,
                initial,
                plan,
                max_iter=config.bp_max_iter,
                tol=config.bp_tol,
                damping=config.bp_damping,
                residual_check_interval=config.bp_residual_check_interval,
                step_residual_gate_factor=(config.bp_step_residual_gate_factor),
                reuse_opposite_cavities=(config.reuse_opposite_bp_cavities),
                active_compaction_ratio=config.bp_active_compaction_ratio,
            )
            bp_info["init"] = init_label
            messages = {key: value[0] for key, value in solved.items()}
            z_clean = _bethe_factors(clean_batch, peps, solved, plan)["Z"][0]

        _store_clean_layer(ctx, clean_cache, layer, peps, messages, z_clean, bp_info)
        previous_messages = messages
        previous_z = z_clean


def build_clean_cache(
    ctx: Any,
    config: SingleSiteConfig | None = None,
    *,
    gate_data_layers: tuple[int, ...] | None = None,
    background_layer_hook: Callable[..., set[tuple[int, str]] | None] | None = None,
    offload_device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Build the clean source seed; later layers are cached with source BP.

    ``gate_data_layers=None`` retains the full second-order cache.  A selected
    tuple is an exact scheduling optimization for first-order source-layer
    probes: every clean PEPS/BP layer is still evolved and cached, while local
    Kraus amplitudes are read only where an error source is actually spawned.
    """
    config = config or SingleSiteConfig()
    config.validate()
    selected_gate_data_layers = (
        None
        if gate_data_layers is None
        else frozenset(int(layer) for layer in gate_data_layers)
    )
    if selected_gate_data_layers is not None and any(
        layer < 0 or layer >= len(ctx.layers) for layer in selected_gate_data_layers
    ):
        raise ValueError("gate_data_layers contains an invalid layer")
    plan = _build_plan(ctx, config)
    dressed = tuple(
        {
            int(gate["gate_idx"]): compute_dressed_kraus(
                gate["kraus_ops"], gate["ideal_unitary"]
            )
            for gate in gates
        }
        for gates in ctx.layers
    )
    clean_cache = {
        "layers": {},
        "plan": plan,
        "dressed": dressed,
        "config": config,
        "gate_data_layers": selected_gate_data_layers,
        "background_layer_hook": background_layer_hook,
        "offload_device": (
            None if offload_device is None else str(torch.device(offload_device))
        ),
        "build_seconds": 0.0,
        "environment": (
            "single_site_bp_gloop_readout"
            if config.local_readout_method == "gloop"
            else "single_site_bp_cluster_readout"
        ),
    }
    started = time.perf_counter()
    _extend_clean_cache(ctx, clean_cache, 0)
    clean_cache["build_seconds"] = time.perf_counter() - started
    return clean_cache


def _batch_peps(
    peps: dict[int, torch.Tensor],
    batch: int,
) -> dict[int, torch.Tensor]:
    return {
        site: value.unsqueeze(0).expand(batch, *value.shape).clone()
        for site, value in peps.items()
    }


def _shape_signature(tensors: dict[int, torch.Tensor]) -> tuple:
    return tuple(
        (site, tuple(int(x) for x in value.shape[1:]))
        for site, value in sorted(tensors.items())
    )


def _source_records(
    ctx: Any,
    clean: dict[str, Any],
    source_layer: int,
    config: SingleSiteConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    filtered = []
    exact_zero_modes = []
    gauge_diagnostics = []
    by_gate = {}
    total_modes = 0
    total_F_weight = 0.0
    discarded_F_weight = 0.0
    gates = ctx.layers[source_layer]
    gate_to_local = {int(gate["gate_idx"]): local for local, gate in enumerate(gates)}
    for gate in gates:
        gate_id = int(gate["gate_idx"])
        data = clean["gate_data"][gate_id]
        original_amplitudes = data["a_modes"].reshape(-1)
        source_modes, source_amplitudes, unitary = _source_gauge_data(
            data, gate_id, config
        )
        absolute_amplitudes = source_amplitudes.abs()
        total_modes += int(source_amplitudes.numel())
        # An exactly zero source amplitude produces an identically zero mixed
        # ket and therefore has no BP fixed point to normalize.  Removing it is
        # exact (its numerator and fidelity weight are both zero), not an
        # amplitude approximation.  Positive user tolerances retain their
        # historical inclusive threshold.
        structurally_nonzero = absolute_amplitudes > 0.0
        keep = (
            structurally_nonzero
            if config.source_amplitude_filter_tol == 0.0
            else absolute_amplitudes >= config.source_amplitude_filter_tol
        )
        kept_ids = torch.nonzero(keep, as_tuple=False).flatten().tolist()
        removed_ids = torch.nonzero(~keep, as_tuple=False).flatten()
        zero_ids = torch.nonzero(~structurally_nonzero, as_tuple=False).flatten()
        exact_zero_modes.extend(
            {"gate": gate_id, "mode": int(mode)}
            for mode in zero_ids.detach().cpu().tolist()
        )
        full_F = float(
            torch.sum(absolute_amplitudes.square(), dtype=torch.float64).item()
        )
        removed_F = (
            float(
                torch.sum(
                    absolute_amplitudes.index_select(0, removed_ids).square(),
                    dtype=torch.float64,
                ).item()
            )
            if removed_ids.numel()
            else 0.0
        )
        total_F_weight += full_F
        discarded_F_weight += removed_F
        by_gate[gate_id] = {
            "total_modes": int(source_amplitudes.numel()),
            "kept_modes": len(kept_ids),
            "filtered_mode_ids": [int(value) for value in removed_ids.tolist()],
            "full_F_weight": full_F,
            "discarded_F_weight": removed_F,
            "discarded_F_fraction": (removed_F / full_F if full_F > 0.0 else None),
        }
        if not kept_ids:
            raise RuntimeError(f"source filter removed every mode of gate {gate_id}")
        for mode in kept_ids:
            records.append(
                {
                    "gate": gate_id,
                    "gate_local": gate_to_local[gate_id],
                    "mode": int(mode),
                    "support": data["support"],
                    "E": source_modes[mode],
                    "a_source": source_amplitudes[mode],
                }
            )
        if removed_ids.numel():
            removed_values = (
                absolute_amplitudes.index_select(0, removed_ids).detach().cpu().tolist()
            )
            filtered.extend(
                {"gate": gate_id, "mode": int(mode), "abs_a": float(value)}
                for mode, value in zip(removed_ids.tolist(), removed_values)
            )
        unitary_error = 0.0
        if unitary is not None:
            identity = torch.eye(
                int(unitary.shape[0]), dtype=unitary.dtype, device=unitary.device
            )
            unitary_error = float(
                torch.max(
                    torch.abs(unitary @ unitary.conj().transpose(-1, -2) - identity)
                ).item()
            )
        gauge_diagnostics.append(
            {
                "gate": gate_id,
                "mode_count": int(source_amplitudes.numel()),
                "original_min_abs_a": float(original_amplitudes.abs().min().item()),
                "original_max_abs_a": float(original_amplitudes.abs().max().item()),
                "rotated_min_abs_a": float(absolute_amplitudes.min().item()),
                "rotated_max_abs_a": float(absolute_amplitudes.max().item()),
                "amplitude_norm_sq_before": float(
                    torch.sum(
                        original_amplitudes.abs().square(), dtype=torch.float64
                    ).item()
                ),
                "amplitude_norm_sq_after": float(
                    torch.sum(
                        source_amplitudes.abs().square(), dtype=torch.float64
                    ).item()
                ),
                "unitary_max_abs_error": unitary_error,
            }
        )
    return records, {
        "enabled": config.source_amplitude_filter_tol > 0,
        "exact_zero_modes_excluded": exact_zero_modes,
        "backend": (
            "single_site_bp_gloop_readout"
            if config.local_readout_method == "gloop"
            else "single_site_bp_cluster_readout"
        ),
        "amplitude_tol": float(config.source_amplitude_filter_tol),
        "total_modes": total_modes,
        "kept_modes": len(records),
        "filtered_modes": filtered,
        "total_F_weight": total_F_weight,
        "discarded_F_weight": discarded_F_weight,
        "discarded_F_fraction": (
            discarded_F_weight / total_F_weight if total_F_weight > 0.0 else None
        ),
        "by_gate": by_gate,
        "source_kraus_gauge": config.source_kraus_gauge,
        "source_kraus_gauge_phase": config.source_kraus_gauge_phase,
        "source_kraus_gauge_seed": config.source_kraus_gauge_seed,
        "gauge_diagnostics": gauge_diagnostics,
    }


def _support_signatures(
    records: list[dict[str, Any]],
    clean_peps: dict[int, torch.Tensor],
    *,
    config: SingleSiteConfig,
    neighbors: dict[int, dict[str, int]],
) -> dict[tuple[int, ...], tuple]:
    first_by_support = {}
    for record in records:
        first_by_support.setdefault(record["support"], record)
    signatures = {}
    for support, record in first_by_support.items():
        probe = _batch_peps(clean_peps, 1)
        dimension = 2 ** len(support)
        _apply_operator_qr(
            probe,
            support,
            record["E"].reshape(1, dimension, dimension),
            chi_max=config.chi_max,
            neighbors=neighbors,
            cache_shared_gate=False,
        )
        signatures[support] = _shape_signature(probe)
    return signatures


def _make_source_tile(
    records: list[dict[str, Any]],
    signature: tuple,
    clean_peps: dict[int, torch.Tensor],
    *,
    first_branch_id: int,
    config: SingleSiteConfig,
    neighbors: dict[int, dict[str, int]],
) -> dict[str, Any]:
    B = len(records)
    ref = next(iter(clean_peps.values()))
    final = {
        site: torch.empty((B, *shape), dtype=ref.dtype, device=ref.device)
        for site, shape in signature
    }
    positions_by_support = {}
    for position, record in enumerate(records):
        positions_by_support.setdefault(record["support"], []).append(position)
    for support, positions in positions_by_support.items():
        selected = [records[position] for position in positions]
        fragment = _batch_peps(clean_peps, len(selected))
        dimension = 2 ** len(support)
        operators = torch.stack([record["E"] for record in selected]).reshape(
            len(selected), dimension, dimension
        )
        _apply_operator_microbatched(
            fragment, support, operators, config=config, neighbors=neighbors
        )
        if _shape_signature(fragment) != signature:
            raise RuntimeError("source tile shape probe disagrees with insertion")
        index = torch.tensor(positions, dtype=torch.long, device=ref.device)
        for site, value in fragment.items():
            final[site].index_copy_(0, index, value)
    return {
        "tensors": final,
        "branch_ids": torch.arange(
            first_branch_id, first_branch_id + B, device=ref.device
        ),
        "source_gate_local": torch.tensor(
            [record["gate_local"] for record in records],
            dtype=torch.long,
            device=ref.device,
        ),
        "source_mode_local": torch.tensor(
            [record["mode"] for record in records], dtype=torch.long, device=ref.device
        ),
        "source_amplitudes": torch.stack([record["a_source"] for record in records])
        .reshape(B)
        .to(device=ref.device),
    }


def _iter_source_tiles(
    records: list[dict[str, Any]],
    clean_peps: dict[int, torch.Tensor],
    *,
    config: SingleSiteConfig,
    neighbors: dict[int, dict[str, int]],
) -> Iterator[dict[str, Any]]:
    signatures = _support_signatures(
        records, clean_peps, config=config, neighbors=neighbors
    )
    buckets: dict[tuple, list[dict[str, Any]]] = {}
    for record in records:
        buckets.setdefault(signatures[record["support"]], []).append(record)
    branch_id = 0
    for signature, bucket in buckets.items():
        for start in range(0, len(bucket), config.branch_batch_size):
            selected = bucket[start : start + config.branch_batch_size]
            yield _make_source_tile(
                selected,
                signature,
                clean_peps,
                first_branch_id=branch_id,
                config=config,
                neighbors=neighbors,
            )
            branch_id += len(selected)


def _can_append_clean_branch(
    tensors: dict[int, torch.Tensor],
    clean_peps: dict[int, torch.Tensor],
) -> bool:
    return tensors.keys() == clean_peps.keys() and all(
        tuple(value.shape[1:]) == tuple(clean_peps[site].shape)
        for site, value in tensors.items()
    )


def _append_clean_branch(
    tensors: dict[int, torch.Tensor],
    clean_peps: dict[int, torch.Tensor],
) -> None:
    for site, value in tuple(tensors.items()):
        tensors[site] = torch.cat((value, clean_peps[site].unsqueeze(0)), dim=0)


def _join_source_clean_messages(
    source: dict[tuple[int, str], torch.Tensor],
    clean: dict[tuple[int, str], torch.Tensor],
) -> dict[tuple[int, str], torch.Tensor]:
    return {key: torch.cat((value, clean[key]), dim=0) for key, value in source.items()}


def _slice_bp_info(
    info: dict[str, Any],
    start: int,
    stop: int,
    *,
    init: str,
) -> dict[str, Any]:
    iterations = info["iterations_per_branch"][start:stop]
    return {
        "iterations": max(iterations, default=0),
        "iterations_per_branch": iterations,
        "residual_per_branch": info["residual_per_branch"][start:stop],
        "projective_raw_residual_per_branch": (
            info["projective_raw_residual_per_branch"][start:stop]
        ),
        "raw_map_residual_per_branch": (
            info["raw_map_residual_per_branch"][start:stop]
        ),
        "step_residual_per_branch": (info["step_residual_per_branch"][start:stop]),
        "strict_residual_checks": int(info.get("strict_residual_checks", 0)),
        "strict_residual_rows": int(info.get("strict_residual_rows", 0)),
        "strict_residual_gate_skips": int(info.get("strict_residual_gate_skips", 0)),
        "active_compactions": int(info.get("active_compactions", 0)),
        "init": init,
    }


def _filter_bp_info(
    info: dict[str, Any],
    positions: list[int],
    *,
    init: str,
) -> dict[str, Any]:
    """Keep BP diagnostics for the source branches retained in a pair mask."""
    fields = (
        "iterations_per_branch",
        "residual_per_branch",
        "projective_raw_residual_per_branch",
        "raw_map_residual_per_branch",
        "step_residual_per_branch",
    )
    filtered = {
        field: [info[field][position] for position in positions] for field in fields
    }
    return {
        "iterations": max(filtered["iterations_per_branch"], default=0),
        **filtered,
        "strict_residual_checks": int(info.get("strict_residual_checks", 0)),
        "strict_residual_rows": int(info.get("strict_residual_rows", 0)),
        "strict_residual_gate_skips": int(info.get("strict_residual_gate_skips", 0)),
        "active_compactions": int(info.get("active_compactions", 0)),
        "init": init,
    }


def _merge_full_warm_messages(
    full_branch: dict[int, torch.Tensor],
    clean_peps: dict[int, torch.Tensor],
    clean_messages: dict[tuple[int, str], torch.Tensor],
    retained_messages: dict[tuple[int, str], torch.Tensor],
    retained_positions: torch.Tensor,
    plan: dict[str, Any],
    *,
    invalidated: set[tuple[int, str]],
) -> tuple[dict[tuple[int, str], torch.Tensor], torch.Tensor]:
    """Reset omitted rows while preserving converged rows in full order."""
    merged, _ = _initialize_messages(
        full_branch,
        clean_peps,
        plan,
        warm=None,
        clean=clean_messages,
        invalidated=invalidated,
    )
    valid = torch.zeros(
        int(next(iter(full_branch.values())).shape[0]),
        dtype=torch.bool,
        device=next(iter(full_branch.values())).device,
    )
    if retained_positions.numel():
        for key, value in retained_messages.items():
            merged[key].index_copy_(0, retained_positions, value)
        valid.index_fill_(0, retained_positions, True)
    return merged, valid


def compute_source_layer(
    ctx: Any,
    clean_cache: dict[str, Any],
    source_layer: int,
    config: SingleSiteConfig | None = None,
    *,
    skip_source_branch_ids_by_target_layer: dict[int, tuple[int, ...]] | None = None,
    auto_skip_nonconverged_pairs: bool | None = None,
    source_amplitudes_override: dict[int, torch.Tensor] | None = None,
    branch_diagnostic_pairs: set[tuple[int, int]] | None = None,
    diagnostic_exact_source_amplitudes: dict[int, torch.Tensor] | None = None,
    readout_modes: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Compute one source layer with streamed single-site mixed BP.

    ``skip_source_branch_ids_by_target_layer`` is an explicit diagnostic
    control: selected source branches are omitted only from the BP/readout at
    the named target layer, while the physical branch tensors still evolve
    through that layer and later targets.  The returned ledger is therefore a
    partial-source approximation at those target layers.

    When ``auto_skip_nonconverged_pairs`` is true, branches that remain active
    after ``bp_max_iter`` are recorded in the same target-layer map and are
    omitted only from that target readout.  Other branches continue normally.

    ``source_amplitudes_override`` is a diagnostic-only anchor replacement.
    The source mode selection still comes from the clean BP cache, while the
    supplied amplitudes are used only in the source-anchored branch weights
    and numerator.

    ``branch_diagnostic_pairs`` records per-source-mode readout conditioning
    for selected (source_gate, target_gate) pairs.  It is intentionally
    opt-in because it materializes a small diagnostic ledger.

    ``readout_modes`` is an ablation-only option.  When it contains more than
    one estimator, the same PEPS branches, BP fixed points, raw transition
    tensors, and Bethe factors are reused for every listed readout mode.
    The legacy return fields continue to describe ``config.branch_weight_mode``;
    the additional results are under ``ablation_results``.
    """
    config = config or clean_cache["config"]
    config.validate()
    if auto_skip_nonconverged_pairs is None:
        auto_skip_nonconverged_pairs = config.auto_skip_nonconverged_pairs
    elif not isinstance(auto_skip_nonconverged_pairs, bool):
        raise ValueError("auto_skip_nonconverged_pairs must be boolean or None")
    if config != clean_cache["config"]:
        raise ValueError(
            "compute_source_layer must use the config that built clean_cache"
        )
    if clean_cache.get("environment") not in {
        "single_site_bp_cluster_readout",
        "single_site_bp_gloop_readout",
    }:
        raise ValueError("clean_cache is not a streamed single-site BP cache")
    if not 0 <= source_layer < len(ctx.layers) - 1:
        raise ValueError("source_layer must have at least one later target")

    allowed_readout_modes = {
        "source_anchored",
        "source_anchored_sumratio",
        "bethe_ratio",
        "bethe_unnormalized",
    }
    requested_modes = tuple(readout_modes or (config.branch_weight_mode,))
    if (
        not requested_modes
        or any(mode not in allowed_readout_modes for mode in requested_modes)
        or len(set(requested_modes)) != len(requested_modes)
    ):
        raise ValueError(f"invalid readout_modes: {requested_modes}")
    if config.branch_weight_mode not in requested_modes:
        # A single-mode call keeps the historical behavior.  For an ablation
        # call, the configured mode must remain one of the returned ledgers so
        # the top-level compatibility fields remain meaningful.
        if readout_modes is not None:
            raise ValueError(
                "config.branch_weight_mode must be included in readout_modes"
            )
    ablation_enabled = len(requested_modes) > 1

    plan = clean_cache["plan"]
    layers = clean_cache["layers"]
    _extend_clean_cache(ctx, clean_cache, source_layer)
    neighbors = plan["neighbors"]
    background_layer_hook = clean_cache.get("background_layer_hook")
    skip_by_target = {
        int(layer): set(int(branch_id) for branch_id in branch_ids)
        for layer, branch_ids in (skip_source_branch_ids_by_target_layer or {}).items()
    }
    source_clean = layers[source_layer]
    device = next(iter(source_clean["peps"].values())).device
    records, filter_info = _source_records(ctx, source_clean, source_layer, config)
    if source_amplitudes_override is not None:
        rotated_overrides = {}
        for gate_id, amplitudes in source_amplitudes_override.items():
            gate_id = int(gate_id)
            value = torch.as_tensor(
                amplitudes,
                dtype=source_clean["peps"][next(iter(source_clean["peps"]))].dtype,
                device=device,
            ).reshape(-1)
            if gate_id not in {
                int(gate["gate_idx"]) for gate in ctx.layers[source_layer]
            }:
                raise ValueError(
                    f"source_amplitudes_override contains unknown gate {gate_id}"
                )
            expected = int(source_clean["gate_data"][gate_id]["a_modes"].numel())
            if int(value.numel()) != expected:
                raise ValueError(
                    f"source amplitude override for gate {gate_id} has "
                    f"{int(value.numel())} modes, expected {expected}"
                )
            if bool((~torch.isfinite(value)).any()):
                raise ValueError(
                    f"source amplitude override for gate {gate_id} is non-finite"
                )
            data = source_clean["gate_data"][gate_id]
            _, _, unitary = _source_gauge_data(data, gate_id, config)
            rotated_overrides[gate_id] = (
                unitary @ value if unitary is not None else value
            )
        for record in records:
            override = rotated_overrides.get(int(record["gate"]))
            if override is None:
                raise ValueError(
                    "source_amplitudes_override must cover every retained source gate"
                )
            record["a_source"] = torch.as_tensor(
                override,
                device=record["E"].device,
                dtype=record["E"].dtype,
            ).reshape(-1)[int(record["mode"])]
        filter_info = dict(filter_info)
        filter_info["anchor_override"] = "exact_one_point_amplitude"
    if diagnostic_exact_source_amplitudes is not None:
        rotated_exact_amplitudes = {}
        for gate in ctx.layers[source_layer]:
            gate_id = int(gate["gate_idx"])
            if gate_id not in diagnostic_exact_source_amplitudes:
                raise ValueError(
                    "diagnostic_exact_source_amplitudes must cover every source gate"
                )
            value = torch.as_tensor(
                diagnostic_exact_source_amplitudes[gate_id],
                dtype=source_clean["gate_data"][gate_id]["a_modes"].dtype,
                device=device,
            ).reshape(-1)
            expected = int(source_clean["gate_data"][gate_id]["a_modes"].numel())
            if int(value.numel()) != expected or bool((~torch.isfinite(value)).any()):
                raise ValueError(
                    f"invalid exact diagnostic amplitudes for gate {gate_id}"
                )
            _, _, unitary = _source_gauge_data(
                source_clean["gate_data"][gate_id], gate_id, config
            )
            rotated_exact_amplitudes[gate_id] = (
                unitary @ value if unitary is not None else value
            )
        diagnostic_exact_source_amplitudes = rotated_exact_amplitudes
    target_layers = list(range(source_layer + 1, len(ctx.layers)))
    one_qubit_reuse_plans = {}
    if (
        config.reuse_one_qubit_readout
        and requested_modes == ("source_anchored",)
        and not auto_skip_nonconverged_pairs
        and branch_diagnostic_pairs is None
        and source_amplitudes_override is None
    ):
        for parent_layer in target_layers[:-1]:
            next_layer = parent_layer + 1
            if skip_by_target.get(parent_layer) or skip_by_target.get(next_layer):
                continue
            reuse_plan = _build_one_qubit_readout_reuse_plan(
                ctx, clean_cache, parent_layer
            )
            if reuse_plan is not None:
                one_qubit_reuse_plans[parent_layer] = reuse_plan
    one_qubit_reused_gate_ids_by_layer = {
        int(reuse_plan["target_layer"]): set(
            int(gate_id) for gate_id in reuse_plan["covered_gate_ids"]
        )
        for reuse_plan in one_qubit_reuse_plans.values()
    }
    one_qubit_reuse_projection_calls = 0
    one_qubit_reuse_projection_rows = 0
    source_gate_ids_by_local = torch.tensor(
        [int(gate["gate_idx"]) for gate in ctx.layers[source_layer]],
        dtype=torch.long,
        device=device,
    )
    source_gate_local_by_id = {
        int(gate["gate_idx"]): local
        for local, gate in enumerate(ctx.layers[source_layer])
    }
    pair_num_by_mode = {
        mode: {
            layer: torch.zeros(
                len(ctx.layers[source_layer]),
                len(ctx.layers[layer]),
                dtype=torch.float64,
                device=device,
            )
            for layer in target_layers
        }
        for mode in requested_modes
    }
    pair_den_by_mode = {
        mode: {
            layer: torch.zeros_like(pair_num_by_mode[mode][layer])
            for layer in target_layers
        }
        for mode in requested_modes
    }
    # This mask is metadata about incomplete pairs only.  It must not remove
    # the pair from the numerator ledger: safe source modes in the same pair
    # remain valid and are accumulated below.
    pair_unresolved_by_mode = {
        mode: {
            layer: torch.zeros(
                len(ctx.layers[source_layer]),
                len(ctx.layers[layer]),
                dtype=torch.bool,
                device=device,
            )
            for layer in target_layers
        }
        for mode in requested_modes
    }
    # Compatibility aliases used by the skip bookkeeping below.
    pair_num = pair_num_by_mode[config.branch_weight_mode]
    pair_den = pair_den_by_mode[config.branch_weight_mode]
    pair_unresolved = pair_unresolved_by_mode[config.branch_weight_mode]
    bp_records = []
    branch_diagnostics = []
    quarantine_ledger = []
    kappa_warning_ledger = []
    fallback_queue = []
    fallback_queue_keys = set()

    def _queue_unresolved_branch(
        target_layer: int,
        target: dict[str, Any],
        branch_id: int,
        reason: str,
    ) -> None:
        """Route one omitted source branch to every target support it serves."""
        branch_id = int(branch_id)
        if not 0 <= branch_id < len(records):
            raise RuntimeError(
                f"invalid unresolved branch id {branch_id}; branch_count={len(records)}"
            )
        record = records[branch_id]
        source_gate = int(record["gate"])
        source_mode = int(record["mode"])
        source_local = source_gate_local_by_id[source_gate]
        for mode in requested_modes:
            pair_unresolved_by_mode[mode][target_layer][source_local].fill_(True)
        support_keys = {
            (
                tuple(int(site) for site in target["supports"][gate_id]),
                tuple(int(site) for site in target["readout_supports"][gate_id]),
            )
            for gate_id in target["gate_ids"]
        }
        for support, readout_support in sorted(support_keys):
            key = (branch_id, int(target_layer), support, readout_support)
            if key in fallback_queue_keys:
                continue
            fallback_queue_keys.add(key)
            target_gate_ids = [
                int(gate_id)
                for gate_id in target["gate_ids"]
                if tuple(int(site) for site in target["supports"][gate_id]) == support
                and tuple(int(site) for site in target["readout_supports"][gate_id])
                == readout_support
            ]
            fallback_queue.append(
                {
                    "branch_id": branch_id,
                    "target_layer": int(target_layer),
                    "source_gate": source_gate,
                    "source_mode": source_mode,
                    "source_support": [int(site) for site in record["support"]],
                    "target_support": [int(site) for site in support],
                    "readout_support": [int(site) for site in readout_support],
                    "target_gate_ids": target_gate_ids,
                    "source_amplitude_abs": float(
                        abs(record["a_source"]).detach().cpu().item()
                    ),
                    "reason": reason,
                    "resolution": "pending",
                }
            )

    support_diagnostics = []
    tile_count = 0
    readout_tiling_stats = {
        "attempts": 0,
        "branch_splits": 0,
        "successful_branch_tiles": [],
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()

    tiles = _iter_source_tiles(
        records, source_clean["peps"], config=config, neighbors=neighbors
    )
    for tile_index, tile in enumerate(tiles):
        tile_count += 1
        B = int(tile["branch_ids"].shape[0])
        needs_clean = any(layer not in layers for layer in target_layers)
        co_batch_clean = (
            tile_index == 0
            and needs_clean
            and _can_append_clean_branch(tile["tensors"], source_clean["peps"])
        )
        clean_slot = None
        clean_previous_messages = None
        clean_previous_z = None
        if co_batch_clean:
            clean_slot = B
            _append_clean_branch(tile["tensors"], source_clean["peps"])
            clean_previous_messages = source_clean["messages"]
            clean_previous_z = source_clean["Z_clean"]
        elif needs_clean:
            # A two-site source can change the virtual shape.  Such a clean
            # state cannot share a tensor batch without zero-padding, so keep
            # the exact pre-existing B=1 path for this uncommon case.
            _extend_clean_cache(ctx, clean_cache, target_layers[-1])

        bp_ranges = [
            (start, min(start + config.bp_branch_chunk_size, B))
            for start in range(0, B, config.bp_branch_chunk_size)
        ]
        warm_by_chunk: list[dict[tuple[int, str], torch.Tensor] | None] = [
            None for _ in bp_ranges
        ]
        warm_valid_by_chunk: list[torch.Tensor | None] = [None for _ in bp_ranges]

        for target_layer in target_layers:
            _emit_stage_profile_event(
                "target_layer",
                "begin",
                target_layer=target_layer,
                tile_index=tile_index,
            )
            gates = ctx.layers[target_layer]
            _emit_stage_profile_event(
                "peps_layer_evolution",
                "begin",
                target_layer=target_layer,
                tile_index=tile_index,
            )
            background_invalidated: set[tuple[int, str]] = set()
            if background_layer_hook is not None:
                returned = background_layer_hook(
                    tile["tensors"],
                    target_layer,
                    config=config,
                    neighbors=neighbors,
                )
                if returned is not None:
                    background_invalidated.update(returned)
            invalidated = _apply_layer(
                tile["tensors"], gates, config=config, neighbors=neighbors
            )
            invalidated.update(background_invalidated)
            _emit_stage_profile_event(
                "peps_layer_evolution",
                "end",
                target_layer=target_layer,
                tile_index=tile_index,
            )
            clean_missing = target_layer not in layers
            if clean_missing:
                if not co_batch_clean or clean_slot is None:
                    raise RuntimeError(
                        f"clean layer {target_layer} was not constructed"
                    )
                clean_peps = {
                    site: value[clean_slot] for site, value in tile["tensors"].items()
                }
            else:
                clean = layers[target_layer]
                clean_peps = clean["peps"]
                if co_batch_clean:
                    clean_previous_messages = clean["messages"]
                    clean_previous_z = clean["Z_clean"]
            target = plan["targets"][target_layer]
            # Residual/background updates and bond projections invalidate the
            # mixed environment even when the intended gates are all 1Q.
            # Reusing it can be wrong with unchanged ranks and can fail later
            # in readout when adaptive ranks change.
            exact_oneq = (
                _is_one_qubit_layer(gates)
                and background_layer_hook is None
                and not invalidated
            )
            reused_gate_ids = one_qubit_reused_gate_ids_by_layer.get(
                target_layer, set()
            )
            readout_gate_ids = tuple(
                int(gate_id)
                for gate_id in target["gate_ids"]
                if int(gate_id) not in reused_gate_ids
            )
            reuse_projection_plan = one_qubit_reuse_plans.get(target_layer)

            for chunk_index, (start, stop) in enumerate(bp_ranges):
                full_branch_ids = tile["branch_ids"][start:stop]
                skip_ids = skip_by_target.get(target_layer, set())
                skipped_positions = [
                    position
                    for position, branch_id in enumerate(
                        full_branch_ids.detach().cpu().tolist()
                    )
                    if int(branch_id) in skip_ids
                ]
                for position in skipped_positions:
                    _queue_unresolved_branch(
                        target_layer,
                        target,
                        int(full_branch_ids[position].item()),
                        "explicit_or_previous_target_skip",
                    )
                active_positions = [
                    position
                    for position in range(stop - start)
                    if position not in set(skipped_positions)
                ]
                if not active_positions:
                    warm_by_chunk[chunk_index] = None
                    warm_valid_by_chunk[chunk_index] = None
                    continue
                active_index = None
                if len(active_positions) != stop - start:
                    active_index = torch.tensor(
                        active_positions,
                        dtype=torch.long,
                        device=full_branch_ids.device,
                    )
                branch = {
                    site: value[start:stop] for site, value in tile["tensors"].items()
                }
                source_local_indices = tile["source_gate_local"][start:stop]
                source_mode_indices = tile["source_mode_local"][start:stop]
                source_amplitudes = tile["source_amplitudes"][start:stop]
                branch_positions = torch.arange(
                    stop - start, dtype=torch.long, device=full_branch_ids.device
                )
                if active_index is not None:
                    branch = {
                        site: value.index_select(0, active_index)
                        for site, value in branch.items()
                    }
                    branch_ids = full_branch_ids.index_select(0, active_index)
                    source_local_indices = source_local_indices.index_select(
                        0, active_index
                    )
                    source_mode_indices = source_mode_indices.index_select(
                        0, active_index
                    )
                    source_amplitudes = source_amplitudes.index_select(0, active_index)
                    branch_positions = branch_positions.index_select(0, active_index)
                else:
                    branch_ids = full_branch_ids
                warm = warm_by_chunk[chunk_index]
                warm_valid = warm_valid_by_chunk[chunk_index]
                if active_index is not None:
                    if warm is not None:
                        warm = {
                            key: value.index_select(0, active_index)
                            for key, value in warm.items()
                        }
                    if warm_valid is not None:
                        warm_valid = warm_valid.index_select(0, active_index)
                warm_is_exact = (
                    warm is not None
                    and warm_valid is not None
                    and bool(warm_valid.all().item())
                )
                local_B = stop - start
                if active_index is not None:
                    local_B = int(active_index.numel())
                include_clean = clean_missing and co_batch_clean and chunk_index == 0
                record_skipped_ids = {
                    int(full_branch_ids[position].item())
                    for position in skipped_positions
                }

                if include_clean:
                    clean_branch = {
                        site: value[clean_slot : clean_slot + 1]
                        for site, value in tile["tensors"].items()
                    }
                    if exact_oneq and warm_is_exact:
                        messages = warm
                        bp_info = {
                            "iterations": 0,
                            "iterations_per_branch": [0] * local_B,
                            "residual_per_branch": [0.0] * local_B,
                            "projective_raw_residual_per_branch": ([0.0] * local_B),
                            "raw_map_residual_per_branch": [0.0] * local_B,
                            "step_residual_per_branch": [0.0] * local_B,
                            "init": "exact_1q_reuse",
                        }
                        clean_messages = clean_previous_messages
                        clean_z = clean_previous_z
                        clean_bp_info = {
                            "iterations": 0,
                            "iterations_per_branch": [0],
                            "residual_per_branch": [0.0],
                            "projective_raw_residual_per_branch": [0.0],
                            "raw_map_residual_per_branch": [0.0],
                            "step_residual_per_branch": [0.0],
                            "init": "exact_1q_reuse_co_batched",
                        }
                        init_label = "exact_1q_reuse"
                    else:
                        source_initial, source_init = _initialize_messages(
                            branch,
                            clean_peps,
                            plan,
                            warm=warm,
                            clean=clean_previous_messages,
                            invalidated=invalidated,
                        )
                        clean_initial, clean_init = _initialize_messages(
                            clean_branch,
                            clean_peps,
                            plan,
                            warm=clean_previous_messages,
                            clean=None,
                            invalidated=invalidated,
                        )
                        combined_branch = {
                            site: torch.cat((value, clean_branch[site]), dim=0)
                            for site, value in branch.items()
                        }
                        combined_initial = _join_source_clean_messages(
                            source_initial, clean_initial
                        )
                        combined_ids = torch.cat(
                            (
                                branch_ids,
                                branch_ids.new_full((1,), -1),
                            )
                        )
                        try:
                            combined_messages, combined_info = _solve_bp(
                                combined_branch,
                                clean_peps,
                                combined_initial,
                                plan,
                                max_iter=config.bp_max_iter,
                                tol=config.bp_tol,
                                damping=config.bp_damping,
                                residual_check_interval=(
                                    config.bp_residual_check_interval
                                ),
                                step_residual_gate_factor=(
                                    config.bp_step_residual_gate_factor
                                ),
                                reuse_opposite_cavities=(
                                    config.reuse_opposite_bp_cavities
                                ),
                                active_compaction_ratio=(
                                    config.bp_active_compaction_ratio
                                ),
                                branch_ids=combined_ids,
                                allow_nonconverged=(auto_skip_nonconverged_pairs),
                            )
                        except RuntimeError as exc:
                            raise RuntimeError(
                                f"source_layer={source_layer}, "
                                f"target_layer={target_layer}, "
                                f"tile={tile_index}, bp_chunk={chunk_index}: "
                                f"{exc}"
                            ) from exc
                        original_local_B = local_B
                        failed_ids = set(
                            int(value)
                            for value in combined_info.get("failed_branch_ids", [])
                        )
                        if -1 in failed_ids:
                            raise RuntimeError(
                                f"source_layer={source_layer}, "
                                f"target_layer={target_layer}, "
                                "clean branch did not converge; automatic "
                                "pair skipping applies only to source branches"
                            )
                        failed_ids.discard(-1)
                        keep_positions = [
                            position
                            for position, branch_id in enumerate(
                                branch_ids.detach().cpu().tolist()
                            )
                            if int(branch_id) not in failed_ids
                        ]
                        if failed_ids:
                            skip_by_target.setdefault(target_layer, set()).update(
                                failed_ids
                            )
                            record_skipped_ids.update(failed_ids)
                            for failed_id in sorted(failed_ids):
                                _queue_unresolved_branch(
                                    target_layer,
                                    target,
                                    failed_id,
                                    "bp_nonconverged",
                                )
                        if len(keep_positions) != original_local_B:
                            keep_index = torch.tensor(
                                keep_positions,
                                dtype=torch.long,
                                device=branch_ids.device,
                            )
                            branch = {
                                site: value.index_select(0, keep_index)
                                for site, value in branch.items()
                            }
                            branch_ids = branch_ids.index_select(0, keep_index)
                            source_local_indices = source_local_indices.index_select(
                                0, keep_index
                            )
                            source_mode_indices = source_mode_indices.index_select(
                                0, keep_index
                            )
                            source_amplitudes = source_amplitudes.index_select(
                                0, keep_index
                            )
                            branch_positions = branch_positions.index_select(
                                0, keep_index
                            )
                            messages = {
                                key: value[:original_local_B].index_select(
                                    0, keep_index
                                )
                                for key, value in combined_messages.items()
                            }
                            bp_info = _filter_bp_info(
                                _slice_bp_info(
                                    combined_info, 0, original_local_B, init=source_init
                                ),
                                keep_positions,
                                init=source_init,
                            )
                            local_B = len(keep_positions)
                        else:
                            messages = {
                                key: value[:local_B]
                                for key, value in combined_messages.items()
                            }
                            bp_info = _slice_bp_info(
                                combined_info, 0, local_B, init=source_init
                            )
                        clean_messages = {
                            key: value[original_local_B]
                            for key, value in combined_messages.items()
                        }
                        init_label = source_init
                        clean_bp_info = _slice_bp_info(
                            combined_info,
                            original_local_B,
                            original_local_B + 1,
                            init=f"{clean_init}_co_batched",
                        )
                        clean_message_batch = {
                            key: value.unsqueeze(0)
                            for key, value in clean_messages.items()
                        }
                        clean_z = _bethe_factors(
                            clean_branch, clean_peps, clean_message_batch, plan
                        )["Z"][0]

                    _store_clean_layer(
                        ctx,
                        clean_cache,
                        target_layer,
                        clean_peps,
                        clean_messages,
                        clean_z,
                        clean_bp_info,
                    )
                    clean_previous_messages = clean_messages
                    clean_previous_z = clean_z
                    clean = layers[target_layer]
                elif exact_oneq and warm_is_exact:
                    messages = warm
                    bp_info = {
                        "iterations": 0,
                        "iterations_per_branch": [0] * local_B,
                        "residual_per_branch": [0.0] * local_B,
                        "projective_raw_residual_per_branch": ([0.0] * local_B),
                        "raw_map_residual_per_branch": [0.0] * local_B,
                        "step_residual_per_branch": [0.0] * local_B,
                    }
                    init_label = "exact_1q_reuse"
                else:
                    initial, init_label = _initialize_messages(
                        branch,
                        clean_peps,
                        plan,
                        warm=warm,
                        clean=clean["messages"],
                        invalidated=invalidated,
                    )
                    try:
                        messages, bp_info = _solve_bp(
                            branch,
                            clean_peps,
                            initial,
                            plan,
                            max_iter=config.bp_max_iter,
                            tol=config.bp_tol,
                            damping=config.bp_damping,
                            residual_check_interval=(config.bp_residual_check_interval),
                            step_residual_gate_factor=(
                                config.bp_step_residual_gate_factor
                            ),
                            reuse_opposite_cavities=(config.reuse_opposite_bp_cavities),
                            active_compaction_ratio=(config.bp_active_compaction_ratio),
                            branch_ids=branch_ids,
                            allow_nonconverged=(auto_skip_nonconverged_pairs),
                        )
                    except RuntimeError as exc:
                        raise RuntimeError(
                            f"source_layer={source_layer}, "
                            f"target_layer={target_layer}, "
                            f"tile={tile_index}, bp_chunk={chunk_index}: "
                            f"{exc}"
                        ) from exc
                    failed_ids = set(
                        int(value) for value in bp_info.get("failed_branch_ids", [])
                    )
                    keep_positions = [
                        position
                        for position, branch_id in enumerate(
                            branch_ids.detach().cpu().tolist()
                        )
                        if int(branch_id) not in failed_ids
                    ]
                    if failed_ids:
                        skip_by_target.setdefault(target_layer, set()).update(
                            failed_ids
                        )
                        record_skipped_ids.update(failed_ids)
                        for failed_id in sorted(failed_ids):
                            _queue_unresolved_branch(
                                target_layer,
                                target,
                                failed_id,
                                "bp_nonconverged",
                            )
                    if len(keep_positions) != local_B:
                        keep_index = torch.tensor(
                            keep_positions, dtype=torch.long, device=branch_ids.device
                        )
                        branch = {
                            site: value.index_select(0, keep_index)
                            for site, value in branch.items()
                        }
                        messages = {
                            key: value.index_select(0, keep_index)
                            for key, value in messages.items()
                        }
                        branch_ids = branch_ids.index_select(0, keep_index)
                        source_local_indices = source_local_indices.index_select(
                            0, keep_index
                        )
                        source_mode_indices = source_mode_indices.index_select(
                            0, keep_index
                        )
                        source_amplitudes = source_amplitudes.index_select(
                            0, keep_index
                        )
                        branch_positions = branch_positions.index_select(0, keep_index)
                        bp_info = _filter_bp_info(
                            bp_info, keep_positions, init=init_label
                        )
                        local_B = len(keep_positions)
                full_warm_branch = {
                    site: value[start:stop] for site, value in tile["tensors"].items()
                }
                if (
                    active_index is None
                    and not record_skipped_ids
                    and local_B == stop - start
                ):
                    warm_by_chunk[chunk_index] = messages
                    warm_valid_by_chunk[chunk_index] = torch.ones(
                        stop - start, dtype=torch.bool, device=full_branch_ids.device
                    )
                else:
                    merged_warm, merged_valid = _merge_full_warm_messages(
                        full_warm_branch,
                        clean_peps,
                        clean["messages"],
                        messages,
                        branch_positions,
                        plan,
                        invalidated=invalidated,
                    )
                    warm_by_chunk[chunk_index] = merged_warm
                    warm_valid_by_chunk[chunk_index] = merged_valid
                if local_B == 0:
                    continue
                bp_records.append(
                    {
                        "target_layer": target_layer,
                        "tile": tile_index,
                        "bp_chunk": chunk_index,
                        "B": local_B,
                        "init": init_label,
                        "iterations": bp_info["iterations_per_branch"],
                        "residuals": bp_info["residual_per_branch"],
                        "projective_raw_residuals": (
                            bp_info["projective_raw_residual_per_branch"]
                        ),
                        "raw_map_residuals": (bp_info["raw_map_residual_per_branch"]),
                        "step_residuals": bp_info["step_residual_per_branch"],
                        "strict_residual_checks": int(
                            bp_info.get("strict_residual_checks", 0)
                        ),
                        "strict_residual_rows": int(
                            bp_info.get("strict_residual_rows", 0)
                        ),
                        "strict_residual_gate_skips": int(
                            bp_info.get("strict_residual_gate_skips", 0)
                        ),
                        "active_compactions": int(bp_info.get("active_compactions", 0)),
                        "invalidated_message_keys": [
                            [int(site), direction]
                            for site, direction in sorted(invalidated)
                        ],
                        "skipped_source_branch_ids": sorted(record_skipped_ids),
                    }
                )

                readout_needs_bethe = (
                    "bethe_ratio" in requested_modes
                    or "bethe_unnormalized" in requested_modes
                    or config.record_support_consistency
                    or branch_diagnostic_pairs is not None
                )
                full_readout_bethe = (
                    _bethe_factors(branch, clean["peps"], messages, plan)
                    if readout_needs_bethe
                    else None
                )

                for read_start in range(
                    0,
                    local_B,
                    config.readout_branch_chunk_size,
                ):
                    read_stop = min(
                        read_start + config.readout_branch_chunk_size, local_B
                    )
                    read_branch = {
                        site: value[read_start:read_stop]
                        for site, value in branch.items()
                    }
                    read_messages = {
                        key: value[read_start:read_stop]
                        for key, value in messages.items()
                    }
                    if not readout_gate_ids and reuse_projection_plan is None:
                        continue
                    readout_result = _read_target_chunk_auto_tiled(
                        read_branch,
                        clean,
                        read_messages,
                        target,
                        plan,
                        source_amplitudes=source_amplitudes[read_start:read_stop],
                        branch_ids=branch_ids[read_start:read_stop],
                        source_gate_ids=source_gate_ids_by_local.index_select(
                            0, source_local_indices[read_start:read_stop]
                        ),
                        source_mode_ids=source_mode_indices[read_start:read_stop],
                        bp_residuals=torch.as_tensor(
                            bp_info["residual_per_branch"],
                            dtype=torch.float64,
                            device=branch_ids.device,
                        )[read_start:read_stop],
                        target_layer=target_layer,
                        branch_diagnostic_pairs=branch_diagnostic_pairs,
                        branch_diagnostics=branch_diagnostics,
                        exact_source_amplitudes=(diagnostic_exact_source_amplitudes),
                        quarantine_ledger=(
                            quarantine_ledger
                            if config.retain_quarantine_ledger
                            else None
                        ),
                        kappa_warning_ledger=(
                            kappa_warning_ledger
                            if config.retain_quarantine_ledger
                            else None
                        ),
                        fallback_queue=(
                            fallback_queue if config.retain_quarantine_ledger else None
                        ),
                        support_diagnostics=(
                            support_diagnostics
                            if config.record_support_consistency
                            else None
                        ),
                        readout_modes=(requested_modes if ablation_enabled else None),
                        bethe_factors_override=(
                            _slice_bethe_factors(
                                full_readout_bethe, read_start, read_stop
                            )
                            if full_readout_bethe is not None
                            else None
                        ),
                        gate_ids_override=readout_gate_ids,
                        one_qubit_reuse_plan=reuse_projection_plan,
                        tiling_stats=readout_tiling_stats,
                    )
                    source_local = source_local_indices[read_start:read_stop]
                    if isinstance(readout_result, dict) and "primary" in readout_result:
                        reuse_num, reuse_den = readout_result["one_qubit_reuse"]
                        reuse_target_layer = int(reuse_projection_plan["target_layer"])
                        pair_num_by_mode["source_anchored"][
                            reuse_target_layer
                        ].index_add_(0, source_local, reuse_num)
                        pair_den_by_mode["source_anchored"][
                            reuse_target_layer
                        ].index_add_(0, source_local, reuse_den)
                        one_qubit_reuse_projection_calls += 1
                        one_qubit_reuse_projection_rows += read_stop - read_start
                        readout_result = readout_result["primary"]
                    if ablation_enabled:
                        contributions_by_mode = readout_result["by_mode"]
                    else:
                        contribution_num, contribution_den, contribution_quarantine = (
                            readout_result
                        )
                        contributions_by_mode = {
                            config.branch_weight_mode: (
                                contribution_num,
                                contribution_den,
                                contribution_quarantine,
                            )
                        }
                    for mode, (
                        mode_num,
                        mode_den,
                        mode_quarantine,
                    ) in contributions_by_mode.items():
                        _emit_stage_profile_event(
                            "pair_accumulation",
                            "begin",
                            target_layer=target_layer,
                            batch_rows=read_stop - read_start,
                        )
                        pair_num_by_mode[mode][target_layer].index_add_(
                            0, source_local, mode_num
                        )
                        pair_den_by_mode[mode][target_layer].index_add_(
                            0, source_local, mode_den
                        )
                        if (
                            config.auto_quarantine_ill_conditioned
                            or mode_quarantine.any()
                        ):
                            # A read chunk contains several Kraus modes from
                            # the same source gate.  Reduce unresolved flags
                            # by OR before writing; index_copy_ with duplicate
                            # row indices would otherwise let the last mode
                            # overwrite an earlier ill-conditioned mode.
                            for source_local_value in torch.unique(source_local):
                                source_local_int = int(source_local_value.item())
                                mode_rows = source_local == source_local_value
                                pair_unresolved_by_mode[mode][target_layer][
                                    source_local_int
                                ] |= mode_quarantine[mode_rows].any(dim=0)
                        _emit_stage_profile_event(
                            "pair_accumulation",
                            "end",
                            target_layer=target_layer,
                            batch_rows=read_stop - read_start,
                        )
            _emit_stage_profile_event(
                "target_layer", "end", target_layer=target_layer, tile_index=tile_index
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    source_gates = ctx.layers[source_layer]
    source_F_full = torch.stack(
        [
            layers[source_layer]["gate_data"][int(gate["gate_idx"])]["F"]
            for gate in source_gates
        ]
    ).to(dtype=torch.float64, device=device)
    pair_C = {}
    pair_S_safe = {}
    pair_unresolved_ledger = {}
    sum_by_target = {}
    total = 0.0
    pair_count = 0
    reported_pair_count = 0
    quarantined_pairs = set()
    for target_layer in target_layers:
        target_gates = ctx.layers[target_layer]
        target_F = torch.stack(
            [
                layers[target_layer]["gate_data"][int(gate["gate_idx"])]["F"]
                for gate in target_gates
            ]
        ).to(dtype=torch.float64, device=device)
        if config.target_amplitude_filter_tol > 0.0:
            target_F = torch.stack(
                [
                    layers[target_layer]["gate_data"][int(gate["gate_idx"])]["F_target"]
                    for gate in target_gates
                ]
            ).to(dtype=torch.float64, device=device)
        numerator = pair_num[target_layer]
        safe_denominator = pair_den[target_layer] * target_F.unsqueeze(0)
        if config.quarantine_granularity == "branch" and config.branch_weight_mode in {
            "source_anchored",
            "bethe_unnormalized",
        }:
            # F_a is the full source-gate fidelity, including unresolved
            # source modes.  The numerator still contains only the stable
            # branch rows until the fallback queue is resolved.
            denominator = source_F_full.unsqueeze(1) * target_F.unsqueeze(0)
        else:
            denominator = safe_denominator
        unresolved = pair_unresolved[target_layer]
        valid = (
            torch.ones_like(unresolved)
            if config.quarantine_granularity == "branch"
            else ~unresolved
        )
        invalid_denominator = (~torch.isfinite(denominator)) | (denominator <= 1e-30)
        if bool((invalid_denominator & valid).any()):
            raise RuntimeError(f"invalid fidelity denominator at layer {target_layer}")
        matrix = torch.zeros_like(numerator)
        if bool(valid.any()):
            matrix[valid] = numerator[valid] / denominator[valid] - 1.0
        if bool((~torch.isfinite(matrix[valid])).any()):
            raise RuntimeError(f"non-finite C2 matrix at layer {target_layer}")
        matrix_cpu = matrix.detach().cpu()
        valid_cpu = valid.detach().cpu()
        layer_sum = float(matrix_cpu[valid_cpu].sum().item())
        sum_by_target[target_layer] = layer_sum
        total += layer_sum
        pair_count += len(source_gates) * len(target_gates)
        reported_pair_count += int(valid.sum().item())
        unresolved_positions = torch.nonzero(unresolved, as_tuple=False).detach().cpu()
        for source_local, target_local in unresolved_positions.tolist():
            pair_unresolved_ledger[
                (
                    int(source_gates[source_local]["gate_idx"]),
                    int(target_gates[target_local]["gate_idx"]),
                )
            ] = {
                "safe_numerator": float(
                    numerator[source_local, target_local].detach().cpu().item()
                ),
                "safe_denominator": float(
                    denominator[source_local, target_local].detach().cpu().item()
                ),
            }
        invalid_positions = torch.nonzero(~valid, as_tuple=False).detach().cpu()
        for source_local, target_local in invalid_positions.tolist():
            quarantined_pairs.add(
                (
                    int(source_gates[source_local]["gate_idx"]),
                    int(target_gates[target_local]["gate_idx"]),
                )
            )
        if config.retain_pair_ledger:
            for source_local, source_gate in enumerate(source_gates):
                for target_local, target_gate in enumerate(target_gates):
                    pair_key = (
                        int(source_gate["gate_idx"]),
                        int(target_gate["gate_idx"]),
                    )
                    pair_S_safe[pair_key] = float(
                        numerator[source_local, target_local].detach().cpu().item()
                    )
            for source_local, source_gate in enumerate(source_gates):
                for target_local, target_gate in enumerate(target_gates):
                    if not bool(valid_cpu[source_local, target_local]):
                        continue
                    pair_C[
                        (
                            int(source_gate["gate_idx"]),
                            int(target_gate["gate_idx"]),
                        )
                    ] = float(matrix_cpu[source_local, target_local].item())

    # The compatibility block above finalizes config.branch_weight_mode using
    # the historical fields.  Finalize any additional ablation modes from
    # their separately accumulated numerators, without rerunning BP.
    ablation_results = {
        config.branch_weight_mode: {
            "sum_C2": total,
            "sum_C2_safe_numerator_known": total,
            "sum_by_target_layer": sum_by_target,
            "pair_count": pair_count,
            "reported_pair_count": reported_pair_count,
            "all_pair_count": pair_count,
            "quarantined_pair_count": len(quarantined_pairs),
            "quarantined_pairs": [
                [int(source), int(target)]
                for source, target in sorted(quarantined_pairs)
            ],
            "pair_C": pair_C,
            "pair_S": pair_S_safe,
            "pair_S_safe": pair_S_safe,
            "pair_unresolved": pair_unresolved_ledger,
            "unresolved_pair_count": len(pair_unresolved_ledger),
        }
    }
    for mode in requested_modes:
        if mode == config.branch_weight_mode:
            continue
        mode_pair_C = {}
        mode_pair_S_safe = {}
        mode_pair_unresolved = {}
        mode_sum_by_target = {}
        mode_total = 0.0
        mode_reported_pair_count = 0
        mode_quarantined_pairs = set()
        for target_layer in target_layers:
            target_gates = ctx.layers[target_layer]
            target_F = torch.stack(
                [
                    layers[target_layer]["gate_data"][int(gate["gate_idx"])]["F"]
                    for gate in target_gates
                ]
            ).to(dtype=torch.float64, device=device)
            if config.target_amplitude_filter_tol > 0.0:
                target_F = torch.stack(
                    [
                        layers[target_layer]["gate_data"][int(gate["gate_idx"])][
                            "F_target"
                        ]
                        for gate in target_gates
                    ]
                ).to(dtype=torch.float64, device=device)
            numerator = pair_num_by_mode[mode][target_layer]
            safe_denominator = pair_den_by_mode[mode][
                target_layer
            ] * target_F.unsqueeze(0)
            if config.quarantine_granularity == "branch" and mode in {
                "source_anchored",
                "bethe_unnormalized",
            }:
                denominator = source_F_full.unsqueeze(1) * target_F.unsqueeze(0)
            else:
                denominator = safe_denominator
            unresolved = pair_unresolved_by_mode[mode][target_layer]
            valid = (
                torch.ones_like(unresolved)
                if config.quarantine_granularity == "branch"
                else ~unresolved
            )
            invalid_denominator = (~torch.isfinite(denominator)) | (
                denominator <= 1e-30
            )
            if bool((invalid_denominator & valid).any()):
                raise RuntimeError(
                    f"invalid fidelity denominator for {mode} at layer {target_layer}"
                )
            matrix = torch.zeros_like(numerator)
            if bool(valid.any()):
                matrix[valid] = numerator[valid] / denominator[valid] - 1.0
            if bool((~torch.isfinite(matrix[valid])).any()):
                raise RuntimeError(
                    f"non-finite C2 matrix for {mode} at layer {target_layer}"
                )
            matrix_cpu = matrix.detach().cpu()
            valid_cpu = valid.detach().cpu()
            layer_sum = float(matrix_cpu[valid_cpu].sum().item())
            mode_sum_by_target[target_layer] = layer_sum
            mode_total += layer_sum
            mode_reported_pair_count += int(valid.sum().item())
            for source_local, target_local in (
                torch.nonzero(unresolved, as_tuple=False).detach().cpu().tolist()
            ):
                key = (
                    int(source_gates[source_local]["gate_idx"]),
                    int(target_gates[target_local]["gate_idx"]),
                )
                mode_pair_unresolved[key] = {
                    "safe_numerator": float(
                        numerator[source_local, target_local].detach().cpu().item()
                    ),
                    "safe_denominator": float(
                        denominator[source_local, target_local].detach().cpu().item()
                    ),
                }
            for source_local, source_gate in enumerate(source_gates):
                for target_local, target_gate in enumerate(target_gates):
                    key = (int(source_gate["gate_idx"]), int(target_gate["gate_idx"]))
                    mode_pair_S_safe[key] = float(
                        numerator[source_local, target_local].detach().cpu().item()
                    )
                    if not bool(valid_cpu[source_local, target_local]):
                        mode_quarantined_pairs.add(key)
                    else:
                        mode_pair_C[key] = float(
                            matrix_cpu[source_local, target_local].item()
                        )
        ablation_results[mode] = {
            "sum_C2": mode_total,
            "sum_C2_safe_numerator_known": mode_total,
            "sum_by_target_layer": mode_sum_by_target,
            "pair_count": pair_count,
            "reported_pair_count": mode_reported_pair_count,
            "all_pair_count": pair_count,
            "quarantined_pair_count": len(mode_quarantined_pairs),
            "quarantined_pairs": [
                [int(source), int(target)]
                for source, target in sorted(mode_quarantined_pairs)
            ],
            "pair_C": mode_pair_C,
            "pair_S": mode_pair_S_safe,
            "pair_S_safe": mode_pair_S_safe,
            "pair_unresolved": mode_pair_unresolved,
            "unresolved_pair_count": len(mode_pair_unresolved),
        }

    target_mode_filter = {
        "enabled": config.target_amplitude_filter_tol > 0.0,
        "amplitude_tol": float(config.target_amplitude_filter_tol),
        "total_modes": sum(
            int(layers[layer]["gate_data"][int(gate["gate_idx"])]["a_modes"].numel())
            for layer in target_layers
            for gate in ctx.layers[layer]
        ),
        "kept_modes": sum(
            int(
                layers[layer]["gate_data"][int(gate["gate_idx"])]["target_mode_keep"]
                .sum()
                .item()
            )
            for layer in target_layers
            for gate in ctx.layers[layer]
        ),
        "discarded_F_weight": sum(
            float(
                layers[layer]["gate_data"][int(gate["gate_idx"])][
                    "target_discarded_F_weight"
                ]
            )
            for layer in target_layers
            for gate in ctx.layers[layer]
        ),
    }

    return {
        "source_layer": source_layer,
        "skipped_source_branch_ids_by_target_layer": {
            int(layer): sorted(int(branch_id) for branch_id in branch_ids)
            for layer, branch_ids in skip_by_target.items()
        },
        "sum_C2": total,
        "sum_C2_safe_numerator_known": total,
        "sum_by_target_layer": sum_by_target,
        "pair_count": pair_count,
        "reported_pair_count": reported_pair_count,
        "all_pair_count": pair_count,
        "quarantined_pair_count": len(quarantined_pairs),
        "quarantined_pairs": [
            [int(source), int(target)] for source, target in sorted(quarantined_pairs)
        ],
        "pair_C": pair_C,
        "pair_S": pair_S_safe,
        "pair_S_safe": pair_S_safe,
        "pair_unresolved": pair_unresolved_ledger,
        "unresolved_pair_count": len(pair_unresolved_ledger),
        "fallback_queue": (fallback_queue if config.retain_quarantine_ledger else []),
        "readout_method": (
            "source_anchored_conditional_ratio_of_sums"
            if config.branch_weight_mode == "source_anchored"
            else (
                "source_anchored_local_sumratio"
                if config.branch_weight_mode == "source_anchored_sumratio"
                else (
                    "bethe_unnormalized_full_bethe"
                    if config.branch_weight_mode == "bethe_unnormalized"
                    else "bethe_ratio_conditional_ratio_of_sums"
                )
            )
        ),
        "branch_weight_mode": config.branch_weight_mode,
        "local_readout_method": config.local_readout_method,
        "gloop_size": (
            config.gloop_size if config.local_readout_method == "gloop" else None
        ),
        "gloop_combine": (
            config.gloop_combine if config.local_readout_method == "gloop" else None
        ),
        "gloop_audit": dict(plan["gloop_audit"]),
        "readout_modes": list(requested_modes),
        "ablation_results": ablation_results if ablation_enabled else {},
        "readout_block": config.readout_block,
        "readout_boundary_mode": config.readout_boundary_mode,
        "readout_contraction_mode": config.readout_contraction_mode,
        "joint_halo_rank": config.joint_halo_rank,
        "joint_halo_rank_audit": dict(plan["joint_halo_rank_audit"]),
        "bounded_2x2_path_audit": dict(plan["bounded_2x2_path_audit"]),
        "readout_scheduling": {
            "auto_tiling": config.auto_readout_tiling,
            "fused_operators_requested": config.fuse_readout_operators,
            "one_qubit_reuse_requested": (config.reuse_one_qubit_readout),
            "one_qubit_reuse_target_layers": sorted(one_qubit_reused_gate_ids_by_layer),
            "one_qubit_reuse_covered_gate_count": sum(
                len(gate_ids)
                for gate_ids in one_qubit_reused_gate_ids_by_layer.values()
            ),
            "one_qubit_reuse_projection_calls": (one_qubit_reuse_projection_calls),
            "one_qubit_reuse_projection_rows": (one_qubit_reuse_projection_rows),
            "raw_conditioning_ledger_retained": (config.retain_quarantine_ledger),
            "requested_branch_chunk": config.readout_branch_chunk_size,
            "patch_batch_size": config.readout_patch_batch_size,
            **readout_tiling_stats,
        },
        "branch_count": len(records),
        "tile_count": tile_count,
        "mode_filter": filter_info,
        "target_mode_filter": target_mode_filter,
        "branch_diagnostics": branch_diagnostics,
        "conditional_kappa": {
            "enabled": config.auto_quarantine_ill_conditioned,
            "warn_threshold": float(config.conditional_kappa_warn),
            "skip_threshold": float(config.conditional_kappa_skip),
            "quarantined_pair_count": len(quarantined_pairs),
            "unresolved_pair_count": len(pair_unresolved_ledger),
            "fallback_queue_count": len(fallback_queue),
            "quarantine_ledger": (
                quarantine_ledger if config.retain_quarantine_ledger else []
            ),
            "warning_ledger": (
                kappa_warning_ledger if config.retain_quarantine_ledger else []
            ),
        },
        "bp_records": bp_records,
        "support_consistency": {
            "enabled": config.record_support_consistency,
            "samples": len(support_diagnostics),
            "max_abs_spread": max(
                (item["max_abs_spread"] for item in support_diagnostics), default=0.0
            ),
            "mean_abs_spread": (
                sum(item["mean_abs_spread"] for item in support_diagnostics)
                / len(support_diagnostics)
                if support_diagnostics
                else 0.0
            ),
        },
        "wall_seconds": elapsed,
        "environment": clean_cache["environment"],
    }


def _pauli_positive_probabilities(local_rhos: torch.Tensor) -> torch.Tensor:
    """Return terminal ``(I +/- sigma)/2`` weights as ``[..., N, 3, 2]``."""
    dtype = local_rhos.dtype
    device = local_rhos.device
    pauli = torch.stack(
        (
            torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device),
            torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device),
            torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device),
        )
    )
    trace = torch.diagonal(local_rhos, dim1=-2, dim2=-1).sum(-1).real
    signed = torch.einsum("aij,...qji->...qa", pauli, local_rhos).real
    return torch.stack(
        (
            (trace[..., :, None] + signed) / 2.0,
            (trace[..., :, None] - signed) / 2.0,
        ),
        dim=-1,
    ).to(torch.float64)


def _summarize_pauli_probabilities(q: torch.Tensor) -> dict[str, Any]:
    value = q.detach().to(torch.float64).cpu()
    survival = value[..., 0] + value[..., 1]
    signed = value[..., 0] - value[..., 1]
    conditional = signed / survival.clamp_min(1e-300)
    return {
        "p_plus": value[..., 0].tolist(),
        "p_minus": value[..., 1].tolist(),
        "survival_by_basis": survival.tolist(),
        "pauli_unconditional": signed.tolist(),
        "pauli_conditional": conditional.tolist(),
        "survival_basis_spread_max": float(
            (survival.max(dim=-1).values - survival.min(dim=-1).values).max()
        ),
    }


def _connected_first_order_pauli(
    q0: torch.Tensor,
    qg: torch.Tensor,
    q0_by_gate: torch.Tensor,
    floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear one-source correction with matched clean/error readout.

    ``qg[g]`` is the terminal effect weight with source gate ``g`` replaced
    by its complete Kraus channel.  A first-order one-source expansion is
    therefore additive at the unnormalized density/effect level:

        q^(1) = q0 + sum_g (qg[g] - q0_by_gate[g]).

    Multiplying the individual ``qg / q0`` ratios separately for every
    terminal effect is a nonlinear resummation.  In particular, it need not
    preserve the common trace shared by the X, Y, and Z terminal bases.
    """
    eligible = (
        torch.isfinite(q0)
        & torch.all(torch.isfinite(q0_by_gate), dim=0)
        & torch.all(torch.isfinite(qg), dim=0)
    )
    q1_add = q0 + (qg - q0_by_gate).sum(dim=0)
    return q1_add, eligible


def _pauli_read_site_rho_unchecked(
    ket: dict[int, torch.Tensor],
    bra: dict[int, torch.Tensor],
    messages: dict[tuple[int, str], torch.Tensor],
    plan: dict[str, Any],
    z_bethe: torch.Tensor,
    site: int,
    config: SingleSiteConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Enqueue one terminal rho and return it with a deferred trace audit."""
    batch = int(next(iter(ket.values())).shape[0])
    bra_is_batched = next(iter(bra.values())).ndim == next(iter(ket.values())).ndim
    rows = []
    device = next(iter(ket.values())).device
    invalid_trace = torch.zeros((), dtype=torch.bool, device=device)
    chunk_size = (
        int(config.readout_branch_chunk_size)
        if bra_is_batched
        else min(int(config.readout_branch_chunk_size), batch)
    )
    audit = plan.get("gloop_audit", {})

    def contract_range(
        start: int,
        stop: int,
        padded_size: int,
        *,
        retry_on_oom: bool = True,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        valid = stop - start
        chunk_invalid = torch.zeros((), dtype=torch.bool, device=device)

        def padded_chunk(value: torch.Tensor) -> torch.Tensor:
            chunk = value[start:stop]
            if valid == padded_size:
                return chunk
            padding = chunk[-1:].expand(padded_size - valid, *chunk.shape[1:])
            return torch.cat((chunk, padding), dim=0)

        # A short final chunk is padded by duplicating its last independent
        # branch. The duplicate outputs are discarded below. This preserves
        # every physical result while keeping one fixed Cotengra expression
        # shape across all branch tiles.
        ket_chunk = {key: padded_chunk(value) for key, value in ket.items()}
        bra_chunk = (
            {key: padded_chunk(value) for key, value in bra.items()}
            if bra_is_batched
            else bra
        )
        message_chunk = {key: padded_chunk(value) for key, value in messages.items()}
        z_chunk = padded_chunk(z_bethe)
        retry_size: int | None = None
        try:
            if plan.get("local_readout_method", "cluster") == "gloop":
                raw, gloop_invalid_trace = _gloop_transition_unchecked(
                    ket_chunk, bra_chunk, message_chunk, (site,), plan
                )
                chunk_invalid = chunk_invalid | gloop_invalid_trace
            else:
                raw = _raw_transition(
                    ket_chunk,
                    bra_chunk,
                    message_chunk,
                    (site,),
                    plan,
                    readout_support=(site,),
                )
        except torch.OutOfMemoryError as error:
            if not retry_on_oom:
                error.__traceback__ = None
                error.__context__ = None
                error.__cause__ = None
                raise
            if padded_size <= 1:
                raise
            retry_size = max(1, padded_size // 2)
            audit["oom_retry_count"] = int(audit.get("oom_retry_count", 0)) + 1
            previous_min = audit.get("oom_retry_min_branch_chunk")
            audit["oom_retry_min_branch_chunk"] = (
                retry_size
                if previous_min is None
                else min(int(previous_min), retry_size)
            )
            print(
                f"[pauli-readout-oom-retry] site={site} "
                f"branches={start}:{stop} failed_chunk={padded_size} "
                f"retry_chunk={retry_size}",
                flush=True,
            )
            # Do not recurse while the OOM handler is active.  Python keeps
            # the current exception and its traceback frames alive until the
            # ``except`` block is exited; those frames can retain large
            # Cotengra/Einsum intermediates.  Retrying inside the handler
            # therefore made the r4, r2 and r1 attempts accumulate GPU memory
            # instead of giving each smaller attempt a clean baseline.
            error.__traceback__ = None
            error.__context__ = None
            error.__cause__ = None

        if retry_size is not None:
            del ket_chunk, bra_chunk, message_chunk, z_chunk
            gc.collect()
            if next(iter(ket.values())).device.type == "cuda":
                torch.cuda.empty_cache()
            recovered: list[torch.Tensor] = []
            recovered_invalid = torch.zeros((), dtype=torch.bool, device=device)
            for retry_start in range(start, stop, retry_size):
                retry_stop = min(retry_start + retry_size, stop)
                retry_rows, retry_invalid = contract_range(
                    retry_start, retry_stop, retry_size
                )
                recovered.extend(retry_rows)
                recovered_invalid = recovered_invalid | retry_invalid
            return recovered, recovered_invalid
        trace = torch.diagonal(raw, dim1=-2, dim2=-1).sum(-1)
        # Defer the Python bool conversion: converting here synchronizes every
        # branch chunk and serializes otherwise independent CUDA streams.
        chunk_invalid = (
            chunk_invalid | ((~torch.isfinite(trace)) | (trace.abs() <= 1e-30)).any()
        )
        normalized = raw / trace[:, None, None] * z_chunk[:, None, None]
        return [normalized[:valid]], chunk_invalid

    ranges = [
        (start, min(start + chunk_size, batch)) for start in range(0, batch, chunk_size)
    ]
    branch_stream_count = min(
        max(1, int(config.readout_branch_stream_count)),
        len(ranges),
    )
    use_branch_streams = (
        branch_stream_count > 1 and torch.cuda.is_available() and device.type == "cuda"
    )
    if not use_branch_streams:
        for start, stop in ranges:
            chunk_rows, chunk_invalid = contract_range(start, stop, chunk_size)
            rows.extend(chunk_rows)
            invalid_trace = invalid_trace | chunk_invalid
        return torch.cat(rows, dim=0), invalid_trace

    current_stream = torch.cuda.current_stream(device=device)
    worker_streams = [
        torch.cuda.Stream(device=device) for _ in range(branch_stream_count)
    ]
    parallel_disabled = False
    for wave_start in range(0, len(ranges), branch_stream_count):
        wave = ranges[wave_start : wave_start + branch_stream_count]
        if parallel_disabled or len(wave) == 1:
            for start, stop in wave:
                chunk_rows, chunk_invalid = contract_range(start, stop, chunk_size)
                rows.extend(chunk_rows)
                invalid_trace = invalid_trace | chunk_invalid
            continue

        pending = []
        parallel_oom = None
        for worker, (start, stop) in zip(worker_streams, wave):
            worker.wait_stream(current_stream)
            try:
                with torch.cuda.stream(worker):
                    chunk_rows, chunk_invalid = contract_range(
                        start,
                        stop,
                        chunk_size,
                        retry_on_oom=False,
                    )
                pending.append((worker, chunk_rows, chunk_invalid))
            except torch.OutOfMemoryError as error:
                parallel_oom = error
                break

        if parallel_oom is not None:
            # Finish and release every already-enqueued stream before falling
            # back. Recompute the complete wave serially so no branch is
            # double-counted or omitted.
            torch.cuda.synchronize(device=device)
            del pending
            parallel_oom.__traceback__ = None
            parallel_oom.__context__ = None
            parallel_oom.__cause__ = None
            del parallel_oom
            gc.collect()
            torch.cuda.empty_cache()
            audit["branch_stream_oom_fallback_count"] = (
                int(audit.get("branch_stream_oom_fallback_count", 0)) + 1
            )
            print(
                f"[pauli-readout-stream-oom-fallback] site={site} "
                f"streams={branch_stream_count}",
                flush=True,
            )
            parallel_disabled = True
            for start, stop in wave:
                chunk_rows, chunk_invalid = contract_range(start, stop, chunk_size)
                rows.extend(chunk_rows)
                invalid_trace = invalid_trace | chunk_invalid
            continue

        for worker, _, _ in pending:
            current_stream.wait_stream(worker)
        for _, chunk_rows, chunk_invalid in pending:
            for rho in chunk_rows:
                rho.record_stream(current_stream)
            chunk_invalid.record_stream(current_stream)
            rows.extend(chunk_rows)
            invalid_trace = invalid_trace | chunk_invalid
    return torch.cat(rows, dim=0), invalid_trace


def _pauli_read_site_rho(
    ket: dict[int, torch.Tensor],
    bra: dict[int, torch.Tensor],
    messages: dict[tuple[int, str], torch.Tensor],
    plan: dict[str, Any],
    z_bethe: torch.Tensor,
    site: int,
    config: SingleSiteConfig,
) -> torch.Tensor:
    """Read one terminal rho and validate its deferred trace audit."""
    rho, invalid_trace = _pauli_read_site_rho_unchecked(
        ket, bra, messages, plan, z_bethe, site, config
    )
    if bool(invalid_trace):
        raise RuntimeError(f"site {site}: zero/non-finite norm belief")
    return rho


def _pauli_read_local_rhos(
    ket: dict[int, torch.Tensor],
    bra: dict[int, torch.Tensor],
    messages: dict[tuple[int, str], torch.Tensor],
    plan: dict[str, Any],
    z_bethe: torch.Tensor,
    n_qubits: int,
    config: SingleSiteConfig,
    *,
    site_indices: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Read all requested one-site beliefs with one shared BP environment."""
    sites = (
        tuple(range(n_qubits))
        if site_indices is None
        else tuple(int(site) for site in site_indices)
    )
    rows = []
    site_chunk_size = min(int(config.readout_site_chunk_size), len(sites))
    use_cuda_streams = (
        site_chunk_size > 1
        and torch.cuda.is_available()
        and next(iter(ket.values())).device.type == "cuda"
    )
    if use_cuda_streams:
        device = next(iter(ket.values())).device
        current_stream = torch.cuda.current_stream(device=device)
        worker_streams = [
            torch.cuda.Stream(device=device) for _ in range(site_chunk_size)
        ]
    else:
        current_stream = None
        worker_streams = []

    for group_start in range(0, len(sites), site_chunk_size):
        group = sites[group_start : group_start + site_chunk_size]
        # Cotengra expressions cache contraction structure only and are safe
        # to reuse across matched clean/error banks. PyTorch's caching
        # allocator should also retain and reuse released intermediate blocks;
        # empty_cache() here would add allocation/synchronization overhead.
        for offset, site in enumerate(group):
            sequence = group_start + offset + 1
            print(
                f"[pauli-readout] site={site} ({sequence}/{len(sites)})",
                flush=True,
            )
        if not use_cuda_streams or len(group) == 1:
            rows.extend(
                _pauli_read_site_rho(ket, bra, messages, plan, z_bethe, site, config)
                for site in group
            )
            continue

        pending = []
        for worker, site in zip(worker_streams, group):
            worker.wait_stream(current_stream)
            with torch.cuda.stream(worker):
                rho, invalid_trace = _pauli_read_site_rho_unchecked(
                    ket, bra, messages, plan, z_bethe, site, config
                )
            pending.append((site, worker, rho, invalid_trace))
        for _, worker, _, _ in pending:
            current_stream.wait_stream(worker)
        for site, _, rho, invalid_trace in pending:
            rho.record_stream(current_stream)
            invalid_trace.record_stream(current_stream)
            if bool(invalid_trace):
                raise RuntimeError(f"site {site}: zero/non-finite norm belief")
            rows.append(rho)
    return torch.stack(rows, dim=1)


def _pauli_batched_bra_site_plan(plan: dict[str, Any]) -> tuple:
    cached = plan.get("pauli_batched_bra_site_plan")
    if cached is not None:
        return cached
    grouped = []
    for site in sorted(plan["neighbors"]):
        entries = []
        for out_dir in "UDLR":
            if out_dir not in plan["neighbors"][site]:
                continue
            fragments = ["z u d l r p", "z U D L R p"]
            incoming = []
            for direction in "UDLR":
                if direction == out_dir or direction not in plan["neighbors"][site]:
                    continue
                other = plan["neighbors"][site][direction]
                incoming.append((other, OPPOSITE[direction]))
                fragments.append("z " + direction.lower() + " " + direction.upper())
            equation = (
                ", ".join(fragments)
                + " -> z "
                + out_dir.lower()
                + " "
                + out_dir.upper()
            )
            entries.append((site, out_dir, equation, tuple(incoming)))
        grouped.append((site, tuple(entries)))
    result = tuple(grouped)
    plan["pauli_batched_bra_site_plan"] = result
    return result


def _initialize_pauli_norm_messages(
    ket: dict[int, torch.Tensor],
    clean_messages: dict[tuple[int, str], torch.Tensor],
    plan: dict[str, Any],
) -> tuple[dict[tuple[int, str], torch.Tensor], str]:
    batch = int(next(iter(ket.values())).shape[0])
    ref = next(iter(ket.values()))
    messages = {}
    used_clean = True
    for key in plan["directed_edges"]:
        site, direction = key
        axis = DIR_TO_AXIS[direction]
        shape = (
            batch,
            int(ket[site].shape[axis + 1]),
            int(ket[site].shape[axis + 1]),
        )
        candidate = clean_messages.get(key)
        value = None
        if candidate is not None:
            if candidate.ndim == 2 and tuple(candidate.shape) == shape[1:]:
                value = candidate.unsqueeze(0).expand(shape).clone()
            elif candidate.ndim == 3 and tuple(candidate.shape) == shape:
                value = candidate.clone()
        if value is None:
            used_clean = False
            value = (
                torch.eye(shape[1], shape[2], dtype=ref.dtype, device=ref.device)
                .unsqueeze(0)
                .expand(shape)
                .clone()
            )
        value, bad, _ = _normalize_message(value)
        if bool(bad.any()):
            raise RuntimeError(f"message {key}: zero/non-finite norm")
        messages[key] = value
    return messages, ("clean" if used_clean else "partial_clean+cold")


def _solve_pauli_norm_bp(
    ket: dict[int, torch.Tensor],
    initial: dict[tuple[int, str], torch.Tensor],
    plan: dict[str, Any],
    config: SingleSiteConfig,
    branch_ids: torch.Tensor,
) -> tuple[dict[tuple[int, str], torch.Tensor], dict[str, Any]]:
    """Strict batched bra-ket self-overlap BP for terminal error branches."""
    messages = initial
    bra_conj = {site: value.conj() for site, value in ket.items()}
    site_plan = _pauli_batched_bra_site_plan(plan)
    batch = int(next(iter(ket.values())).shape[0])
    device = next(iter(ket.values())).device
    last_projective = torch.full(
        (batch,), float("inf"), dtype=torch.float64, device=device
    )
    last_raw = torch.full_like(last_projective, float("inf"))
    last_step = torch.full_like(last_projective, float("inf"))
    strict_checks = 0
    for iteration in range(1, config.bp_max_iter + 1):
        step = torch.zeros(batch, dtype=torch.float64, device=device)
        for site, entries in site_plan:
            raw_by_key = _contract_bp_site_messages(
                ket,
                bra_conj,
                messages,
                site,
                entries,
                plan,
                reuse_opposite_cavities=False,
            )
            for _, direction, _, _ in entries:
                key = (site, direction)
                old = messages[key]
                raw, bad_raw, _ = _normalize_message(raw_by_key[key])
                if bool(bad_raw.any()):
                    ids = branch_ids[bad_raw].detach().cpu().tolist()
                    raise RuntimeError(
                        f"zero/non-finite norm BP map for branches {ids}"
                    )
                raw = _phase_align(raw, old)
                updated = (1.0 - config.bp_damping) * raw + config.bp_damping * old
                updated, bad_updated, _ = _normalize_message(updated)
                if bool(bad_updated.any()):
                    ids = branch_ids[bad_updated].detach().cpu().tolist()
                    raise RuntimeError(
                        f"zero/non-finite norm BP update for branches {ids}"
                    )
                step = torch.maximum(
                    step,
                    (updated - old)
                    .abs()
                    .reshape(batch, -1)
                    .max(dim=1)
                    .values.to(torch.float64),
                )
                messages[key] = updated
        scheduled = (
            iteration == 1
            or iteration % config.bp_residual_check_interval == 0
            or iteration == config.bp_max_iter
        )
        if not scheduled:
            continue
        strict_checks += 1
        projective, raw_map, bad = _strict_bp_map_residual(
            ket, bra_conj, messages, site_plan, plan, reuse_opposite_cavities=False
        )
        if bool(bad.any()):
            ids = branch_ids[bad].detach().cpu().tolist()
            raise RuntimeError(f"zero/non-finite strict norm BP map for branches {ids}")
        last_projective, last_raw, last_step = projective, raw_map, step
        if bool((projective < config.bp_tol).all()):
            return messages, {
                "iterations": iteration,
                "iterations_per_branch": [iteration] * batch,
                "residual_per_branch": projective.detach().cpu().tolist(),
                "projective_raw_residual_per_branch": (
                    projective.detach().cpu().tolist()
                ),
                "raw_map_residual_per_branch": raw_map.detach().cpu().tolist(),
                "step_residual_per_branch": step.detach().cpu().tolist(),
                "strict_residual_checks": strict_checks,
                "active_compactions": 0,
                "batched_bra_extension": True,
            }
    worst = int(last_projective.argmax().item())
    raise RuntimeError(
        "bra-ket self-overlap norm BP failed after "
        f"{config.bp_max_iter} sweeps; branch_id={int(branch_ids[worst])}, "
        f"projective_residual={float(last_projective[worst]):.3e}, "
        f"raw_map_residual={float(last_raw[worst]):.3e}, "
        f"step_residual={float(last_step[worst]):.3e}"
    )


def _solve_pauli_norm_environment(
    ket: dict[int, torch.Tensor],
    clean_messages: dict,
    plan: dict,
    config: SingleSiteConfig,
    branch_ids: torch.Tensor,
) -> tuple[dict, dict, torch.Tensor, dict[int, torch.Tensor]]:
    initial, init_label = _initialize_pauli_norm_messages(ket, clean_messages, plan)
    messages, info = _solve_pauli_norm_bp(ket, initial, plan, config, branch_ids)
    info["init"] = init_label
    factors = _bethe_factors(ket, ket, messages, plan)
    z = factors["Z"].real.to(torch.float64)
    if bool(((~torch.isfinite(z)) | (z <= 0)).any()):
        raise RuntimeError(
            "non-positive/non-finite bra-ket self-overlap Bethe partition"
        )
    return messages, info, z, ket


def _clean_virtual_bond_audit(clean: dict[str, Any]) -> dict[str, Any]:
    layer_stats = []
    for layer, data in sorted(clean["layers"].items()):
        maximum = max(
            max(int(value) for value in tensor.shape[:4])
            for tensor in data["peps"].values()
        )
        layer_stats.append({"layer": int(layer), "max_virtual_bond": maximum})
    final_layer = max(clean["layers"])
    final_shapes = {
        int(site): [int(value) for value in tensor.shape[:4]]
        for site, tensor in sorted(clean["layers"][final_layer]["peps"].items())
    }
    return {
        "per_layer": layer_stats,
        "final_layer": int(final_layer),
        "final_virtual_shapes_by_site": final_shapes,
        "final_max_virtual_bond": max(max(shape) for shape in final_shapes.values()),
    }


def _build_final_clean_environment(
    ctx: Any,
    config: SingleSiteConfig,
    *,
    background_layer_hook: Callable[..., set[tuple[int, str]] | None] | None = None,
) -> dict[str, Any]:
    """Evolve and solve only the final clean environment needed for q0.

    ``build_clean_cache`` additionally computes every intermediate gate's
    clean Kraus amplitudes because first- and second-order source expansions
    need them.  An ideal-only q0 scan does not.  Skipping those intermediate
    local readouts leaves PEPS evolution and BP equations unchanged while
    avoiding thousands of irrelevant gloop contractions.
    """
    plan = _build_plan(ctx, config)
    neighbors = plan["neighbors"]
    peps = _zero_peps(ctx)
    previous_messages = None
    previous_z = None
    layer_stats = []
    final_bp_info = None
    progress_enabled = os.environ.get("NCS_CLEAN_BACKGROUND_PROGRESS", "0") == "1"
    background_started = time.perf_counter()

    def report_progress(layer: int, stage: str) -> None:
        if not progress_enabled:
            return
        memory = ""
        if torch.device(ctx.device).type == "cuda":
            memory = (
                f" allocated_gib={torch.cuda.memory_allocated(ctx.device) / 1024**3:.3f}"
                f" reserved_gib={torch.cuda.memory_reserved(ctx.device) / 1024**3:.3f}"
            )
        print(
            f"[clean-background] layer={layer + 1}/{len(ctx.layers)}"
            f" stage={stage} elapsed_sec={time.perf_counter() - background_started:.1f}"
            f"{memory}",
            flush=True,
        )

    for layer, gates in enumerate(ctx.layers):
        batched = {site: value.unsqueeze(0) for site, value in peps.items()}
        # This final-only builder does not need an immutable previous-layer
        # snapshot.  Let replaced tensors be freed as each gate updates them.
        del peps
        background_invalidated: set[tuple[int, str]] = set()
        if background_layer_hook is not None:
            report_progress(layer, "residual_start")
            returned = background_layer_hook(
                batched,
                layer,
                config=config,
                neighbors=neighbors,
            )
            if returned is not None:
                background_invalidated.update(returned)
        report_progress(layer, "intended_gates_start")
        invalidated = _apply_layer(batched, gates, config=config, neighbors=neighbors)
        invalidated.update(background_invalidated)
        peps = {site: value[0] for site, value in batched.items()}
        clean_batch = {site: value.unsqueeze(0) for site, value in peps.items()}
        report_progress(layer, "bp_start")
        if (
            previous_messages is not None
            and _is_one_qubit_layer(gates)
            and background_layer_hook is None
        ):
            messages = previous_messages
            z_clean = previous_z
            bp_info = {
                "iterations": 0,
                "iterations_per_branch": [0],
                "residual_per_branch": [0.0],
                "projective_raw_residual_per_branch": [0.0],
                "raw_map_residual_per_branch": [0.0],
                "step_residual_per_branch": [0.0],
                "init": "exact_1q_reuse",
            }
        else:
            initial, init_label = _initialize_messages(
                clean_batch,
                peps,
                plan,
                warm=previous_messages,
                clean=None,
                invalidated=invalidated,
            )
            solved, bp_info = _solve_bp(
                clean_batch,
                peps,
                initial,
                plan,
                max_iter=config.bp_max_iter,
                tol=config.bp_tol,
                damping=config.bp_damping,
                residual_check_interval=config.bp_residual_check_interval,
                step_residual_gate_factor=(config.bp_step_residual_gate_factor),
                reuse_opposite_cavities=(config.reuse_opposite_bp_cavities),
                active_compaction_ratio=config.bp_active_compaction_ratio,
            )
            bp_info["init"] = init_label
            messages = {key: value[0] for key, value in solved.items()}
            z_clean = _bethe_factors(clean_batch, peps, solved, plan)["Z"][0]
        maximum = max(
            max(int(value) for value in tensor.shape[:4]) for tensor in peps.values()
        )
        layer_stats.append(
            {
                "layer": int(layer),
                "max_virtual_bond": maximum,
                "bp_iterations": int(bp_info["iterations"]),
                "bp_max_residual": max(
                    float(value) for value in bp_info["residual_per_branch"]
                ),
            }
        )
        previous_messages = messages
        previous_z = z_clean
        final_bp_info = bp_info
        del batched, clean_batch
        report_progress(layer, "complete")

    final_shapes = {
        int(site): [int(value) for value in tensor.shape[:4]]
        for site, tensor in sorted(peps.items())
    }
    return {
        "peps": peps,
        "messages": previous_messages,
        "Z_clean": previous_z,
        "plan": plan,
        "bp_info": final_bp_info,
        "bond_audit": {
            "per_layer": layer_stats,
            "final_layer": len(ctx.layers) - 1,
            "final_virtual_shapes_by_site": final_shapes,
            "final_max_virtual_bond": max(
                max(shape) for shape in final_shapes.values()
            ),
        },
    }


def compute_ideal_pauli(
    ctx: Any,
    config: SingleSiteConfig | None = None,
    *,
    site_indices: tuple[int, ...] | None = None,
    background_layer_hook: Callable[..., set[tuple[int, str]] | None] | None = None,
) -> dict[str, Any]:
    """Compute terminal ideal local Pauli values without error branches.

    This is the production q0 entry point for local-readout convergence
    scans.  It builds the identical clean PEPS and BP cache used by the
    one-error and two-error engines, then changes only the terminal local
    contraction according to ``config.local_readout_method``.
    """
    config = config or SingleSiteConfig()
    config.validate()
    selected_sites = (
        tuple(range(int(ctx.n_qubits)))
        if site_indices is None
        else tuple(sorted({int(site) for site in site_indices}))
    )
    if not selected_sites:
        raise ValueError("site_indices must select at least one site")
    if any(site < 0 or site >= int(ctx.n_qubits) for site in selected_sites):
        raise ValueError("site_indices contains an invalid site")
    if not ctx.layers:
        raise ValueError("ideal Pauli readout requires at least one layer")

    device = torch.device(ctx.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    clean_started = time.perf_counter()
    clean = _build_final_clean_environment(
        ctx,
        config,
        background_layer_hook=background_layer_hook,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    clean_seconds = time.perf_counter() - clean_started
    clean_ket = {site: value.unsqueeze(0) for site, value in clean["peps"].items()}
    clean_messages = {
        key: value.unsqueeze(0) for key, value in clean["messages"].items()
    }
    clean_z = clean["Z_clean"].real.reshape(1).to(torch.float64)
    read_started = time.perf_counter()
    local_rhos = _pauli_read_local_rhos(
        clean_ket,
        clean["peps"],
        clean_messages,
        clean["plan"],
        clean_z,
        int(ctx.n_qubits),
        config,
        site_indices=selected_sites,
    )[0]
    q0 = _pauli_positive_probabilities(local_rhos)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = int(torch.cuda.max_memory_allocated(device))
        reserved = int(torch.cuda.max_memory_reserved(device))
    else:
        peak = reserved = 0
    return {
        "definition": {
            "connected_order": 0,
            "terminal_effects": "positive (I +/- sigma)/2 effects",
            "reported_expectation": (
                "conditional on global computational-subspace survival"
            ),
            "backend": "clean PEPS + custom BP",
            "terminal_readout": config.local_readout_method,
            "gloop_size": (
                config.gloop_size if config.local_readout_method == "gloop" else None
            ),
        },
        "ideal": _summarize_pauli_probabilities(q0),
        "raw": {
            "site_indices": list(selected_sites),
            "q0": q0.detach().cpu().tolist(),
        },
        "timing": {
            "shared_clean_cache_seconds": clean_seconds,
            "ideal_pauli_readout_seconds": (time.perf_counter() - read_started),
            "total_wall_seconds": time.perf_counter() - started,
        },
        "diagnostics": {
            "clean_bond_audit": clean["bond_audit"],
            "final_bp_info": clean["bp_info"],
            "peak_allocated_bytes": peak,
            "peak_allocated_gib": peak / 1024.0**3,
            "peak_reserved_bytes": reserved,
            "peak_reserved_gib": reserved / 1024.0**3,
            "gloop_audit": dict(clean["plan"]["gloop_audit"]),
        },
    }


def compute_ideal_pauli_gloop_scan(
    ctx: Any,
    config: SingleSiteConfig | None = None,
    *,
    gloop_sizes: tuple[int, ...],
    site_indices: tuple[int, ...] | None = None,
    background_layer_hook: Callable[..., set[tuple[int, str]] | None] | None = None,
    background_checkpoint: Path | None = None,
    background_metadata: dict[str, Any] | None = None,
    scan_checkpoint_callback: Callable[[int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Read several gloop sizes from one evolved clean PEPS/BP background."""
    config = config or SingleSiteConfig(local_readout_method="gloop")
    config.validate()
    if config.local_readout_method != "gloop":
        raise ValueError("gloop scan requires local_readout_method='gloop'")
    sizes = tuple(dict.fromkeys(int(size) for size in gloop_sizes))
    if not sizes or any(size < 1 for size in sizes):
        raise ValueError("gloop_sizes must contain positive integers")
    selected_sites = (
        tuple(range(int(ctx.n_qubits)))
        if site_indices is None
        else tuple(sorted({int(site) for site in site_indices}))
    )
    if not selected_sites:
        raise ValueError("site_indices must select at least one site")
    if any(site < 0 or site >= int(ctx.n_qubits) for site in selected_sites):
        raise ValueError("site_indices contains an invalid site")
    if not ctx.layers:
        raise ValueError("ideal Pauli readout requires at least one layer")

    device = torch.device(ctx.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    clean_started = time.perf_counter()
    reused_background = False
    if background_checkpoint is not None and Path(background_checkpoint).exists():
        from residual_tn.backend.gloop_checkpoints import load_background

        clean = load_background(
            Path(background_checkpoint), background_metadata, device
        )
        clean["plan"] = _build_plan(ctx, config)
        reused_background = True
        print(f"[background-checkpoint] loaded {background_checkpoint}", flush=True)
    else:
        clean = _build_final_clean_environment(
            ctx, config, background_layer_hook=background_layer_hook
        )
        if background_checkpoint is not None:
            from residual_tn.backend.gloop_checkpoints import save_background

            save_background(Path(background_checkpoint), clean, background_metadata)
            print(f"[background-checkpoint] saved {background_checkpoint}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    clean_seconds = time.perf_counter() - clean_started
    clean_ket = {site: value.unsqueeze(0) for site, value in clean["peps"].items()}
    clean_messages = {
        key: value.unsqueeze(0) for key, value in clean["messages"].items()
    }
    clean_z = clean["Z_clean"].real.reshape(1).to(torch.float64)

    scans: dict[str, Any] = {}
    for size in sizes:
        print(f"[gloop-scan] START size={size}", flush=True)
        scan_config = replace(config, gloop_size=int(size))
        scan_config.validate()
        scan_plan = _build_plan(ctx, scan_config)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        read_started = time.perf_counter()
        local_rhos = _pauli_read_local_rhos(
            clean_ket,
            clean["peps"],
            clean_messages,
            scan_plan,
            clean_z,
            int(ctx.n_qubits),
            scan_config,
            site_indices=selected_sites,
        )[0]
        q0 = _pauli_positive_probabilities(local_rhos)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak = int(torch.cuda.max_memory_allocated(device))
            reserved = int(torch.cuda.max_memory_reserved(device))
        else:
            peak = reserved = 0
        scans[str(size)] = {
            "definition": {
                "connected_order": 0,
                "terminal_effects": "positive (I +/- sigma)/2 effects",
                "reported_expectation": (
                    "conditional on global computational-subspace survival"
                ),
                "backend": "shared clean PEPS + custom BP",
                "terminal_readout": "gloop",
                "gloop_size": int(size),
            },
            "ideal": _summarize_pauli_probabilities(q0),
            "raw": {
                "site_indices": list(selected_sites),
                "q0": q0.detach().cpu().tolist(),
            },
            "timing": {
                "ideal_pauli_readout_seconds": (time.perf_counter() - read_started),
            },
            "diagnostics": {
                "peak_allocated_bytes": peak,
                "peak_allocated_gib": peak / 1024.0**3,
                "peak_reserved_bytes": reserved,
                "peak_reserved_gib": reserved / 1024.0**3,
                "gloop_audit": dict(scan_plan["gloop_audit"]),
            },
        }
        del local_rhos, q0, scan_plan
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if scan_checkpoint_callback is not None:
            scan_checkpoint_callback(int(size), scans[str(size)])
        print(f"[gloop-scan] COMPLETE size={size}", flush=True)

    return {
        "definition": {
            "connected_order": 0,
            "backend": "one shared clean PEPS/BP background",
            "terminal_readout": "gloop size convergence scan",
            "gloop_sizes": list(sizes),
            "background_checkpoint_reused": reused_background,
        },
        "shared": {
            "timing": {
                "clean_background_seconds": clean_seconds,
                "total_wall_seconds": time.perf_counter() - started,
            },
            "diagnostics": {
                "clean_bond_audit": clean["bond_audit"],
                "final_bp_info": clean["bp_info"],
            },
        },
        "scans": scans,
    }


def compute_first_order_pauli(
    ctx: Any,
    config: SingleSiteConfig | None = None,
    *,
    reference_floor: float = 1e-10,
    source_layers: tuple[int, ...] | None = None,
    site_indices: tuple[int, ...] | None = None,
    q0_override: torch.Tensor | None = None,
    layer_checkpoint_callback: (Callable[[dict[str, Any]], None] | None) = None,
    background_layer_hook: Callable[..., set[tuple[int, str]] | None] | None = None,
    clean_cache_offload_device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Compute terminal Pauli values for the full one-error Kraus expansion.

    The existing streamed engine still forms and propagates every one-error
    PEPS branch and solves its bra-ket self-overlap BP environment.  Only the
    one-site readout is selected by ``config.local_readout_method``.
    ``source_layers`` and ``site_indices`` are explicit validation controls.
    ``layer_checkpoint_callback`` receives a cumulative, JSON-serializable
    result after each completed source layer. This preserves one shared clean
    history in a long production process while still allowing layer-granular
    recovery.
    """
    config = config or SingleSiteConfig()
    config.validate()
    if reference_floor <= 0:
        raise ValueError("reference_floor must be positive")
    if config.source_amplitude_filter_tol != 0.0:
        raise ValueError(
            "first-order Pauli production forbids nonzero source filtering"
        )
    n_layers = len(ctx.layers)
    selected_layers = (
        tuple(range(n_layers))
        if source_layers is None
        else tuple(sorted({int(layer) for layer in source_layers}))
    )
    if any(layer < 0 or layer >= n_layers for layer in selected_layers):
        raise ValueError("source_layers contains an invalid layer")
    selected_sites = (
        tuple(range(int(ctx.n_qubits)))
        if site_indices is None
        else tuple(sorted({int(site) for site in site_indices}))
    )
    if any(site < 0 or site >= int(ctx.n_qubits) for site in selected_sites):
        raise ValueError("site_indices contains an invalid site")

    device = torch.device(ctx.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    total_started = time.perf_counter()
    clean_started = time.perf_counter()
    clean = build_clean_cache(
        ctx,
        config,
        gate_data_layers=selected_layers,
        background_layer_hook=background_layer_hook,
        offload_device=clean_cache_offload_device,
    )
    _extend_clean_cache(ctx, clean, n_layers - 1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    clean_seconds = time.perf_counter() - clean_started
    plan = clean["plan"]
    final_cached = clean["layers"][n_layers - 1]
    final_clean = {
        **final_cached,
        "messages": {
            key: value.to(device=device)
            for key, value in final_cached["messages"].items()
        },
        "Z_clean": final_cached["Z_clean"].to(device=device),
    }
    if q0_override is None:
        final_clean = _move_clean_layer_tensors(final_cached, device)
        clean_ket = {
            site: value.unsqueeze(0) for site, value in final_clean["peps"].items()
        }
        clean_messages = {
            key: value.unsqueeze(0) for key, value in final_clean["messages"].items()
        }
        clean_z = final_clean["Z_clean"].real.reshape(1).to(torch.float64)
        q0_rho = _pauli_read_local_rhos(
            clean_ket,
            final_clean["peps"],
            clean_messages,
            plan,
            clean_z,
            int(ctx.n_qubits),
            config,
            site_indices=selected_sites,
        )[0]
        q0 = _pauli_positive_probabilities(q0_rho)
        del clean_ket, clean_messages, clean_z
        # Source propagation needs only the converged final clean messages.
        # Release the hydrated multi-GiB final PEPS before loading any source
        # layer from the CPU-offloaded clean history.
        final_clean["peps"].clear()
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        q0 = q0_override.to(device=device, dtype=torch.float64)
        expected_q0_shape = (len(selected_sites), 3, 2)
        if tuple(q0.shape) != expected_q0_shape:
            raise ValueError(
                "q0_override shape mismatch: "
                f"expected {expected_q0_shape}, got {tuple(q0.shape)}"
            )
        if bool((~torch.isfinite(q0)).any()) or bool((q0 <= 0.0).any()):
            raise ValueError("q0_override must be finite and strictly positive")

    selected_gate_ids = [
        int(gate["gate_idx"]) for layer in selected_layers for gate in ctx.layers[layer]
    ]
    gate_to_position = {
        gate_id: position for position, gate_id in enumerate(selected_gate_ids)
    }
    ref = next(iter(final_clean["messages"].values()))
    qg_rho = torch.zeros(
        (len(selected_gate_ids), len(selected_sites), 2, 2),
        dtype=ref.dtype,
        device=ref.device,
    )
    branch_bp_seconds = 0.0
    readout_seconds = 0.0
    branch_count = 0
    max_residual = 0.0
    max_iterations = 0
    source_layer_summaries = []
    for source_layer in selected_layers:
        layer_started = time.perf_counter()
        source_clean = _move_clean_layer_tensors(clean["layers"][source_layer], device)
        records, filter_info = _source_records(ctx, source_clean, source_layer, config)
        layer_branches = 0
        layer_tiles = 0

        def prepare_source_tile(
            tile_index: int,
            tile: dict[str, Any],
        ) -> dict[str, Any]:
            nonlocal branch_bp_seconds, branch_count
            nonlocal layer_branches, layer_tiles
            nonlocal max_residual, max_iterations
            tile_storage_bytes = sum(
                int(value.numel()) * int(value.element_size())
                for value in tile["tensors"].values()
            )
            print(
                f"[source-tile] source_layer={source_layer} "
                f"tile={tile_index} "
                f"branches={int(tile['branch_ids'].numel())} "
                f"peps_storage_gib={tile_storage_bytes / 1024.0**3:.3f}",
                flush=True,
            )
            branch_started = time.perf_counter()
            branch = tile["tensors"]
            for layer in range(source_layer + 1, n_layers):
                if background_layer_hook is not None:
                    background_layer_hook(
                        branch,
                        layer,
                        config=config,
                        neighbors=plan["neighbors"],
                    )
                _apply_layer(
                    branch,
                    ctx.layers[layer],
                    config=config,
                    neighbors=plan["neighbors"],
                )
            messages, info, z, branch_bra = _solve_pauli_norm_environment(
                branch, final_clean["messages"], plan, config, tile["branch_ids"]
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            branch_bp_seconds += time.perf_counter() - branch_started
            max_residual = max(
                max_residual, *(float(value) for value in info["residual_per_branch"])
            )
            max_iterations = max(
                max_iterations, *(int(value) for value in info["iterations_per_branch"])
            )
            local_gate_indices = tile["source_gate_local"].detach().cpu().tolist()
            gate_positions = torch.tensor(
                [
                    gate_to_position[int(ctx.layers[source_layer][local]["gate_idx"])]
                    for local in local_gate_indices
                ],
                dtype=torch.long,
                device=device,
            )
            tile_branches = int(tile["branch_ids"].numel())
            branch_count += tile_branches
            layer_branches += tile_branches
            layer_tiles += 1
            live_tensors = (
                tuple(branch.values())
                + tuple(branch_bra.values())
                + tuple(messages.values())
                + (z,)
            )
            live_bytes = sum(
                int(value.numel()) * int(value.element_size()) for value in live_tensors
            )
            del local_gate_indices, info, tile, live_tensors
            return {
                "tile_index": int(tile_index),
                "branch": branch,
                "branch_bra": branch_bra,
                "messages": messages,
                "z": z,
                "gate_positions": gate_positions,
                "branch_count": tile_branches,
                "live_bytes": live_bytes,
            }

        def release_source_tile(
            bundle: dict[str, Any],
            *,
            final: bool,
        ) -> None:
            tile_index = int(bundle["tile_index"])
            bundle.clear()
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
                label = (
                    "source-tile-cleanup"
                    if final
                    else "source-tile-intermediate-cleanup"
                )
                print(
                    f"[{label}] source_layer={source_layer} "
                    f"tile={tile_index} "
                    f"allocated_gib="
                    f"{torch.cuda.memory_allocated(device) / 1024.0**3:.3f} "
                    f"reserved_gib="
                    f"{torch.cuda.memory_reserved(device) / 1024.0**3:.3f}",
                    flush=True,
                )

        tile_generator = enumerate(
            _iter_source_tiles(
                records,
                source_clean["peps"],
                config=config,
                neighbors=plan["neighbors"],
            ),
            start=1,
        )
        if config.pauli_readout_schedule == "tile_major":
            for tile_index, tile in tile_generator:
                bundle = prepare_source_tile(tile_index, tile)
                read_started = time.perf_counter()
                local_rhos = _pauli_read_local_rhos(
                    bundle["branch"],
                    bundle["branch_bra"],
                    bundle["messages"],
                    plan,
                    bundle["z"],
                    int(ctx.n_qubits),
                    config,
                    site_indices=selected_sites,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                readout_seconds += time.perf_counter() - read_started
                qg_rho.index_add_(0, bundle["gate_positions"], local_rhos)
                del local_rhos
                release_source_tile(bundle, final=True)
        else:
            # Build the complete propagated source bank once.  All targets
            # then reuse it, and one fixed target exhausts every source tile
            # before the next target begins.
            tile_bank = [
                prepare_source_tile(tile_index, tile)
                for tile_index, tile in tile_generator
            ]
            bank_live_bytes = sum(int(bundle["live_bytes"]) for bundle in tile_bank)
            print(
                f"[source-layer-bank] source_layer={source_layer} "
                f"tiles={len(tile_bank)} branches={layer_branches} "
                f"live_storage_gib={bank_live_bytes / 1024.0**3:.3f}",
                flush=True,
            )
            for site_position, site in enumerate(selected_sites):
                print(
                    f"[pauli-target-major] source_layer={source_layer} "
                    f"site={site} "
                    f"({site_position + 1}/{len(selected_sites)})",
                    flush=True,
                )
                read_started = time.perf_counter()
                for bundle in tile_bank:
                    site_rhos = _pauli_read_site_rho(
                        bundle["branch"],
                        bundle["branch_bra"],
                        bundle["messages"],
                        plan,
                        bundle["z"],
                        int(site),
                        config,
                    )
                    qg_rho[:, site_position].index_add_(
                        0, bundle["gate_positions"], site_rhos
                    )
                    del site_rhos
                    # Shape-signature tiles can leave incompatible allocation
                    # classes in the CUDA cache. Drop only free intermediates;
                    # the live propagated source bank remains resident.
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                readout_seconds += time.perf_counter() - read_started
                if device.type == "cuda":
                    print(
                        f"[pauli-target-major-cleanup] "
                        f"source_layer={source_layer} site={site} "
                        f"allocated_gib="
                        f"{torch.cuda.memory_allocated(device) / 1024.0**3:.3f} "
                        f"reserved_gib="
                        f"{torch.cuda.memory_reserved(device) / 1024.0**3:.3f}",
                        flush=True,
                    )
            for bundle in tile_bank:
                release_source_tile(bundle, final=True)
            del tile_bank
        source_layer_summaries.append(
            {
                "source_layer": int(source_layer),
                "gate_count": len(ctx.layers[source_layer]),
                "retained_mode_count": layer_branches,
                "internal_tile_count": layer_tiles,
                "wall_seconds": time.perf_counter() - layer_started,
                "mode_filter": filter_info,
                "completed_before_next_source": True,
            }
        )
        del records, source_clean
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if layer_checkpoint_callback is not None:
            completed_gate_count = sum(
                len(ctx.layers[layer])
                for layer in selected_layers[: len(source_layer_summaries)]
            )
            completed_gate_ids = selected_gate_ids[:completed_gate_count]
            checkpoint_qg = _pauli_positive_probabilities(qg_rho[:completed_gate_count])
            checkpoint_q0_by_gate = q0.unsqueeze(0).expand(
                completed_gate_count, *q0.shape
            )
            checkpoint_q1, checkpoint_eligible = _connected_first_order_pauli(
                q0,
                checkpoint_qg,
                checkpoint_q0_by_gate,
                reference_floor,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                checkpoint_peak = int(torch.cuda.max_memory_allocated(device))
                checkpoint_reserved = int(torch.cuda.max_memory_reserved(device))
            else:
                checkpoint_peak = checkpoint_reserved = 0
            layer_checkpoint_callback(
                {
                    "definition": {
                        "connected_order": 1,
                        "terminal_effects": "positive (I +/- sigma)/2 effects",
                        "reported_expectation": (
                            "conditional on global computational-subspace survival"
                        ),
                        "backend": (
                            "streamed one-error PEPS + bra-ket self-overlap BP"
                        ),
                        "terminal_readout": config.local_readout_method,
                        "gloop_size": (
                            config.gloop_size
                            if config.local_readout_method == "gloop"
                            else None
                        ),
                        "mode_filtering": "exact-zero modes only",
                        "source_unit": "complete circuit layer",
                        "validation_subset": True,
                        "q0_source": (
                            "matched external checkpoint"
                            if q0_override is not None
                            else "current clean PEPS/BP readout"
                        ),
                        "cumulative_layer_checkpoint": True,
                    },
                    "ideal": _summarize_pauli_probabilities(q0),
                    "first_order": _summarize_pauli_probabilities(checkpoint_q1),
                    "raw": {
                        "site_indices": list(selected_sites),
                        "source_layers": [
                            int(row["source_layer"]) for row in source_layer_summaries
                        ],
                        "source_gate_ids": completed_gate_ids,
                        "q0": q0.detach().cpu().tolist(),
                        "qg": checkpoint_qg.detach().cpu().tolist(),
                        "q1": checkpoint_q1.detach().cpu().tolist(),
                        "log_eligible": (checkpoint_eligible.detach().cpu().tolist()),
                    },
                    "timing": {
                        "shared_clean_cache_seconds": clean_seconds,
                        "pauli_branch_environment_seconds": branch_bp_seconds,
                        "all_requested_pauli_readout_seconds": readout_seconds,
                        "total_wall_seconds": (time.perf_counter() - total_started),
                    },
                    "source_layer_schedule": [
                        dict(row) for row in source_layer_summaries
                    ],
                    "diagnostics": {
                        "gate_count": completed_gate_count,
                        "branch_count": sum(
                            int(row["retained_mode_count"])
                            for row in source_layer_summaries
                        ),
                        "clean_bond_audit": _clean_virtual_bond_audit(clean),
                        "max_bp_residual": max_residual,
                        "max_bp_iterations": max_iterations,
                        "peak_allocated_bytes": checkpoint_peak,
                        "peak_allocated_gib": (checkpoint_peak / 1024.0**3),
                        "peak_reserved_bytes": checkpoint_reserved,
                        "peak_reserved_gib": (checkpoint_reserved / 1024.0**3),
                        "gloop_audit": dict(plan["gloop_audit"]),
                    },
                }
            )

    qg = _pauli_positive_probabilities(qg_rho)
    q0_by_gate = q0.unsqueeze(0).expand(len(selected_gate_ids), *q0.shape)
    q1, eligible = _connected_first_order_pauli(q0, qg, q0_by_gate, reference_floor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = int(torch.cuda.max_memory_allocated(device))
        reserved = int(torch.cuda.max_memory_reserved(device))
    else:
        peak = reserved = 0
    return {
        "definition": {
            "connected_order": 1,
            "terminal_effects": "positive (I +/- sigma)/2 effects",
            "reported_expectation": (
                "conditional on global computational-subspace survival"
            ),
            "backend": ("streamed one-error PEPS + bra-ket self-overlap BP"),
            "terminal_readout": config.local_readout_method,
            "gloop_size": (
                config.gloop_size if config.local_readout_method == "gloop" else None
            ),
            "mode_filtering": "exact-zero modes only",
            "source_unit": "complete circuit layer",
            "validation_subset": (
                source_layers is not None or site_indices is not None
            ),
            "q0_source": (
                "matched external checkpoint"
                if q0_override is not None
                else "current clean PEPS/BP readout"
            ),
        },
        "ideal": _summarize_pauli_probabilities(q0),
        "first_order": _summarize_pauli_probabilities(q1),
        "raw": {
            "site_indices": list(selected_sites),
            "source_layers": list(selected_layers),
            "source_gate_ids": selected_gate_ids,
            "q0": q0.detach().cpu().tolist(),
            "qg": qg.detach().cpu().tolist(),
            "q1": q1.detach().cpu().tolist(),
            "log_eligible": eligible.detach().cpu().tolist(),
        },
        "timing": {
            "shared_clean_cache_seconds": clean_seconds,
            "pauli_branch_environment_seconds": branch_bp_seconds,
            "all_requested_pauli_readout_seconds": readout_seconds,
            "total_wall_seconds": time.perf_counter() - total_started,
        },
        "source_layer_schedule": source_layer_summaries,
        "diagnostics": {
            "gate_count": len(selected_gate_ids),
            "branch_count": branch_count,
            "clean_bond_audit": _clean_virtual_bond_audit(clean),
            "max_bp_residual": max_residual,
            "max_bp_iterations": max_iterations,
            "peak_allocated_bytes": peak,
            "peak_allocated_gib": peak / 1024.0**3,
            "peak_reserved_bytes": reserved,
            "peak_reserved_gib": reserved / 1024.0**3,
            "gloop_audit": dict(plan["gloop_audit"]),
        },
    }


def compute_total(
    ctx: Any,
    config: SingleSiteConfig | None = None,
    *,
    background_layer_hook: Callable[..., set[tuple[int, str]] | None] | None = None,
) -> dict[str, Any]:
    """Build one clean cache and sum all source_layer < target_layer pairs."""
    config = config or SingleSiteConfig()
    config.validate()
    device = torch.device(ctx.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    clean_cache = build_clean_cache(
        ctx,
        config,
        background_layer_hook=background_layer_hook,
    )
    source_results = {}
    pair_C = {}
    pair_S_safe = {}
    quarantined_pairs = set()
    unresolved_pairs = set()
    fallback_queue = []
    total = 0.0
    pair_count = 0
    reported_pair_count = 0
    for source_layer in range(len(ctx.layers) - 1):
        result = compute_source_layer(ctx, clean_cache, source_layer, config)
        source_results[source_layer] = result
        total += result["sum_C2"]
        pair_count += result["pair_count"]
        reported_pair_count += result.get("reported_pair_count", result["pair_count"])
        quarantined_pairs.update(
            tuple(pair) for pair in result.get("quarantined_pairs", [])
        )
        unresolved_pairs.update(
            tuple(pair) for pair in result.get("pair_unresolved", {}).keys()
        )
        fallback_queue.extend(result.get("fallback_queue", []))
        if config.retain_pair_ledger:
            overlap = pair_C.keys() & result["pair_C"].keys()
            if overlap:
                raise RuntimeError(f"duplicate pair keys: {tuple(overlap)[:3]}")
            pair_C.update(result["pair_C"])
            pair_S_safe.update(result.get("pair_S_safe", {}))
    expected = sum(
        len(ctx.layers[source]) * len(ctx.layers[target])
        for source in range(len(ctx.layers) - 1)
        for target in range(source + 1, len(ctx.layers))
    )
    if pair_count != expected:
        raise RuntimeError(f"pair_count {pair_count} != expected {expected}")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = int(torch.cuda.max_memory_allocated(device))
    else:
        peak = 0
    return {
        "sum_C2": total,
        "pair_count": pair_count,
        "reported_pair_count": reported_pair_count,
        "quarantined_pair_count": len(quarantined_pairs),
        "quarantined_pairs": [
            [int(source), int(target)] for source, target in sorted(quarantined_pairs)
        ],
        "unresolved_pair_count": len(unresolved_pairs),
        "unresolved_pairs": [
            [int(source), int(target)] for source, target in sorted(unresolved_pairs)
        ],
        "fallback_queue": fallback_queue,
        "pair_C": pair_C,
        "pair_S_safe": pair_S_safe,
        "per_source": source_results,
        "clean_cache_seconds": clean_cache["build_seconds"],
        "total_wall_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": peak,
        "dtype": str(ctx.dtype),
        "device": str(ctx.device),
        "environment": clean_cache["environment"],
        "readout_method": (
            "source_anchored_conditional_ratio_of_sums"
            if config.branch_weight_mode == "source_anchored"
            else (
                "source_anchored_local_sumratio"
                if config.branch_weight_mode == "source_anchored_sumratio"
                else (
                    "bethe_unnormalized_full_bethe"
                    if config.branch_weight_mode == "bethe_unnormalized"
                    else "bethe_ratio_conditional_ratio_of_sums"
                )
            )
        ),
        "branch_weight_mode": config.branch_weight_mode,
        "local_readout_method": config.local_readout_method,
        "gloop_size": (
            config.gloop_size if config.local_readout_method == "gloop" else None
        ),
        "gloop_combine": (
            config.gloop_combine if config.local_readout_method == "gloop" else None
        ),
        "gloop_audit": dict(clean_cache["plan"]["gloop_audit"]),
        "readout_block": config.readout_block,
        "readout_boundary_mode": config.readout_boundary_mode,
        "readout_contraction_mode": config.readout_contraction_mode,
        "joint_halo_rank": config.joint_halo_rank,
        "config": config,
    }


__all__ = [
    "SingleSiteConfig",
    "build_overlap_balancing_unitary",
    "rotate_kraus_modes",
    "build_clean_cache",
    "compute_source_layer",
    "compute_ideal_pauli",
    "compute_first_order_pauli",
    "compute_total",
]
