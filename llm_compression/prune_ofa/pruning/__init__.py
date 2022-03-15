# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Pruning capabilities forked from https://github.com/IntelLabs/Model-Compression-Research-Package/
"""

from .registry import list_methods, list_schedulers
from .schedulers import *
from .methods import *
