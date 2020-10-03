# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

-include ../../makefile.inc


TEST = bslib


test: $(TEST)

%: %.cpp ./bslib_index.cpp ./VQ_quantizer.cpp ./LQ_quantizer.cpp ./PQ_quantizer.cpp ./quantizer.cpp ./utils/utils.h ./HNSWlib/hnswalg.cpp ../../libfaiss.a 
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) -I../../.. $(LIBS)

clean:
	rm -f $(TEST)

.PHONY: all file clean
