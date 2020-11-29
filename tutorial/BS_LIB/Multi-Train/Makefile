

-include ../../../makefile.inc


TEST = faiss_kmeans

test: $(TEST)

faiss_kmeans: $(faiss_kmeans)


%: %.cpp ../bslib_index.cpp ../VQ_quantizer.cpp ../LQ_quantizer.cpp ../PQ_quantizer.cpp ../quantizer.cpp ../utils/utils.h ../HNSWlib/hnswalg.cpp ../../../libfaiss.a 
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) -I../../../.. $(LIBS)

clean:
	rm -f $(TEST)

.PHONY: all TEST file clean
