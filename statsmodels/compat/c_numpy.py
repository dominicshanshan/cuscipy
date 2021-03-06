"""Compatibility functions for cupy versions in lib

np_new_unique
-------------
Optionally provides the count of the number of occurrences of each
unique element.

Copied from cupy source, under license:

Copyright (c) 2005-2015, cupy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the following
  disclaimer in the documentation and/or other materials provided
  with the distribution.

* Neither the name of the cupy Developers nor the names of any
  contributors may be used to endorse or promote products derived
  from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import cupy as cp


# np_matrix_rank = cp.linalg.matrix_rank
# np_new_unique = cp.unique


def recarray_select(recarray, fields):
    """"
    Work-around for changes in cupy 1.13 that return views for recarray
    multiple column selection
    """
    from pandas import DataFrame
    fields = [fields] if not isinstance(fields, (tuple, list)) else fields
    if len(fields) == len(recarray.dtype):
        selection = recarray
    else:
        recarray = DataFrame.from_records(recarray)
        selection = recarray[fields].to_records(index=False)

    return selection


def lstsq(a, b, rcond=None):
    """
    Shim that allows modern rcond setting with backward compat for cupy
    earlier than 1.14
    """
    return cp.linalg.lstsq(a, b, rcond=rcond)
