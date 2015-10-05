INSTALL_PATH?=/usr/local/CTBangBang/
SCRIPT_PATH?=/usr/bin/
SRC_DIR=$(shell pwd)/

all: rebuild

rebuild:
	mkdir -p src/obj
	$(MAKE) -C src ../test
	cp test ctbangbang

install:
ifneq ($(USER),root)
	@echo Please run with sudo
else
	@echo Source directory is: ${SRC_DIR}.
	@echo By default, the CTBB executable and support files
	@echo are installed in:
	@echo 
	@echo ${INSTALL_PATH}
	@echo
	@echo A script, ctbb_recon, will placed in ${SCRIPT_PATH} that should be used
	@echo that should be used to execute reconstructions.
	@echo
	@echo After installation, type "ctbb_recon --help" for more information.

	mkdir -p ${INSTALL_PATH}
	cp -v -u -r ${SRC_DIR}include ${INSTALL_PATH}
	cp -v -u -r ${SRC_DIR}prms ${INSTALL_PATH}
	cp -v -u -r ${SRC_DIR}resources ${INSTALL_PATH}
	cp -v -u -r ${SRC_DIR}src ${INSTALL_PATH}
	cp -v -u -r ${SRC_DIR}ctbangbang ${INSTALL_PATH}
	cp -v -u -r ${SRC_DIR}README.md ${INSTALL_PATH}
	cp -v -u -r ${SRC_DIR}makefile ${INSTALL_PATH}
	cp -v -u -r ${SRC_DIR}LICENSE.md ${INSTALL_PATH}

# Create our execution script
	@echo Creating execution script ${SCRIPT_PATH}ctbb_recon...
	touch ${SCRIPT_PATH}ctbb_recon
	chmod +x ${SCRIPT_PATH}ctbb_recon
	@echo "#!/bin/bash" > ${SCRIPT_PATH}ctbb_recon
	@echo ${INSTALL_PATH}ctbangbang \"\$$\@\" >> ${SCRIPT_PATH}ctbb_recon
	@echo notify-send \"Reconstruction completed\" >> ${SCRIPT_PATH}ctbb_recon
endif


.PHONY: all clean

clean:
	$(MAKE) -C src clean
