

-include ../../../../makefile.inc


TEST = kmeans_PQ

test: $(TEST)

kmeans_PQ: $(kmeans_PQ)


%: %.cpp ../../bslib_index.cpp ../../VQ_quantizer.cpp ../../LQ_quantizer.cpp ../../PQ_quantizer.cpp ../../quantizer.cpp ../../utils/utils.h ../../HNSWlib/hnswalg.cpp ../../../../libfaiss.a 
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) -I../../../../.. $(LIBS)

clean:
	rm -f $(TEST)

.PHONY: all TEST file clean
