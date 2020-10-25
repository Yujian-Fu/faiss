

-include ../../../../makefile.inc


TEST = inverted_index IMI ICI VQTree IVFADC

test: $(TEST)

inverted_index: $(inverted_index)

IMI: $(IMI)

ICI: $(ICI)

VQTree: $(VQTree)

IVFADC: $(IVFADC)

%: %.cpp ../../bslib_index.cpp ../../VQ_quantizer.cpp ../../LQ_quantizer.cpp ../../PQ_quantizer.cpp ../../quantizer.cpp ../../utils/utils.h ../../HNSWlib/hnswalg.cpp ../../../../libfaiss.a 
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) -I../../../../.. $(LIBS)

clean:
	rm -f $(TEST)

.PHONY: all TEST file clean
