"""
Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Gym bindings wrapper module
"""

from __future__ import print_function, division, absolute_import

import importlib
import json
import sys
import os

# from . import gymdeps


def _format_path(pathstr):
    if os.name == "nt":
        # need to flip backslashes and convert to lower case
        return pathstr.replace("\\", "/").lower()
    else:
        return pathstr