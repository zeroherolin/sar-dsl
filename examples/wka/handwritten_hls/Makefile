VITIS_HLS ?= vitis_hls
TIMEOUT ?= timeout
MODULE_TIMEOUT ?= 3600
TOP_TIMEOUT ?= 7200
ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
WORK_DIR := $(ROOT)/work
REPORT_DIR := $(ROOT)/reports
HLS_SCRIPT := $(ROOT)/hls/run_hls.tcl

.PHONY: luts reference-smoke csim-smoke cosim-smoke csynth-corner csynth-row csynth-top reports clean

luts:
	cd "$(ROOT)" && python3 scripts/generate_luts.py

$(WORK_DIR)/logs:
	mkdir -p "$(WORK_DIR)/logs" "$(REPORT_DIR)"

reference-smoke: | $(WORK_DIR)/logs
	python3 "$(ROOT)/reference/wka_reference.py" --n 64 --output "$(REPORT_DIR)/reference_smoke_64.bin"

csim-smoke: luts reference-smoke | $(WORK_DIR)/logs
	cd "$(WORK_DIR)/logs" && WKA_TB_DUMP_PATH="$(REPORT_DIR)/csim_smoke_64.bin" $(TIMEOUT) $(MODULE_TIMEOUT) $(VITIS_HLS) "$(HLS_SCRIPT)" csim smoke wka_sar_top

cosim-smoke: luts reference-smoke | $(WORK_DIR)/logs
	cd "$(WORK_DIR)/logs" && $(TIMEOUT) $(TOP_TIMEOUT) $(VITIS_HLS) "$(HLS_SCRIPT)" cosim smoke wka_sar_top

csynth-corner: luts | $(WORK_DIR)/logs
	cd "$(WORK_DIR)/logs" && $(TIMEOUT) $(MODULE_TIMEOUT) $(VITIS_HLS) "$(HLS_SCRIPT)" csynth production corner_turn_top

csynth-row: luts | $(WORK_DIR)/logs
	cd "$(WORK_DIR)/logs" && $(TIMEOUT) $(MODULE_TIMEOUT) $(VITIS_HLS) "$(HLS_SCRIPT)" csynth production row_transform_top

csynth-top: luts | $(WORK_DIR)/logs
	cd "$(WORK_DIR)/logs" && $(TIMEOUT) $(TOP_TIMEOUT) $(VITIS_HLS) "$(HLS_SCRIPT)" csynth production wka_sar_top

reports:
	python3 "$(ROOT)/scripts/collect_hls_reports.py" --work "$(WORK_DIR)" --output "$(REPORT_DIR)/hls_summary"

clean:
	@test -n "$(ROOT)" -a "$(ROOT)" != "/" -a "$(WORK_DIR)" = "$(ROOT)/work"
	rm -rf "$(WORK_DIR)"
	rm -f "$(REPORT_DIR)"/*.bin "$(REPORT_DIR)"/*.bmp "$(REPORT_DIR)/hls_summary.json" "$(REPORT_DIR)/hls_summary.csv"
	rm -f "$(ROOT)/generated/wka_luts.h"
