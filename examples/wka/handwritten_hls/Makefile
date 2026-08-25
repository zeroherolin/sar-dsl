VITIS_HLS ?= vitis_hls
TIMEOUT ?= timeout
HLS_CSIM_TIMEOUT ?= 86400
COSIM_TIMEOUT ?= 7200
SYNTH_TIMEOUT ?= 7200
ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
WORK_DIR := $(ROOT)/work
REPORT_DIR := $(ROOT)/reports
HLS_SCRIPT := $(ROOT)/hls/run_hls.tcl

.PHONY: luts reference_cosim_256 hls_csim hls_csim_production cosim cosim_medium \
	csynth csynth_corner csynth_row csynth_top reports clean

luts:
	cd "$(ROOT)" && python3 scripts/generate_luts.py

$(WORK_DIR)/logs:
	mkdir -p "$(WORK_DIR)/logs" "$(REPORT_DIR)"

reference_cosim_256: | $(WORK_DIR)/logs
	python3 "$(ROOT)/reference/wka_reference.py" --n 256 --output "$(REPORT_DIR)/reference_cosim_256.bin"

hls_csim: hls_csim_production

hls_csim_production: luts | $(WORK_DIR)/logs
	cd "$(WORK_DIR)/logs" && $(TIMEOUT) $(HLS_CSIM_TIMEOUT) $(VITIS_HLS) "$(HLS_SCRIPT)" hls_csim production wka_sar_top

cosim: cosim_medium

cosim_medium: luts reference_cosim_256 | $(WORK_DIR)/logs
	cd "$(WORK_DIR)/logs" && $(TIMEOUT) $(COSIM_TIMEOUT) $(VITIS_HLS) "$(HLS_SCRIPT)" cosim medium wka_sar_top

csynth: csynth_top

csynth_corner: luts | $(WORK_DIR)/logs
	cd "$(WORK_DIR)/logs" && $(TIMEOUT) $(SYNTH_TIMEOUT) $(VITIS_HLS) "$(HLS_SCRIPT)" csynth production corner_turn_top

csynth_row: luts | $(WORK_DIR)/logs
	cd "$(WORK_DIR)/logs" && $(TIMEOUT) $(SYNTH_TIMEOUT) $(VITIS_HLS) "$(HLS_SCRIPT)" csynth production row_forward_top

csynth_top: luts | $(WORK_DIR)/logs
	cd "$(WORK_DIR)/logs" && $(TIMEOUT) $(SYNTH_TIMEOUT) $(VITIS_HLS) "$(HLS_SCRIPT)" csynth production wka_sar_top

reports:
	python3 "$(ROOT)/scripts/collect_hls_reports.py" --work "$(WORK_DIR)" --output "$(REPORT_DIR)/hls_summary"

clean:
	@test -n "$(ROOT)" -a "$(ROOT)" != "/" -a "$(WORK_DIR)" = "$(ROOT)/work"
	rm -rf "$(WORK_DIR)"
	rm -f "$(REPORT_DIR)"/*.bin "$(REPORT_DIR)"/*.bmp "$(REPORT_DIR)/hls_summary.json" "$(REPORT_DIR)/hls_summary.csv"
	rm -f "$(ROOT)/generated/wka_luts.h"
