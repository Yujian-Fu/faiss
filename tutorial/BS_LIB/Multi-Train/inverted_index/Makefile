

-include ../../../../makefile.inc


TEST = inverted_index_PQ

test: $(TEST)

inverted_index_PQ: $(inverted_index_PQ)

%: %.cpp ../../bslib_index.cpp ../../VQ_quantizer.cpp ../../LQ_quantizer.cpp ../../PQ_quantizer.cpp ../../quantizer.cpp ../../utils/utils.h ../../HNSWlib/hnswalg.cpp ../../../../libfaiss.a 
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) -I../../../../.. $(LIBS)

clean:
	rm -f $(TEST)

.PHONY: all TEST file clean
