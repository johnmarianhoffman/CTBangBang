all: rebuild

rebuild:
	$(MAKE) -C src ../test

.PHONY: all clean

clean:
	$(MAKE) -C src clean
