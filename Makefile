CC = gcc
CFLAGS = -O2 -Wall -std=c99
# Use 32MB stack (recursive build_topo needs deep recursion for autograd graph)
LDFLAGS = -Wl,--stack,33554432 -lm

microgpt: microgpt.c microgpt.h
	$(CC) $(CFLAGS) -o $@ microgpt.c $(LDFLAGS)

clean:
	rm -f microgpt microgpt.exe

.PHONY: clean
