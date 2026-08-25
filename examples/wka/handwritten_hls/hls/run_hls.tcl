# SAR-DSL hand-written omega-K HLS runner.
# Usage: vitis_hls hls/run_hls.tcl <hls_csim|csynth|cosim> <medium|production> ?top?

set script_dir [file dirname [file normalize [info script]]]
set root_dir [file dirname $script_dir]

proc usage {} {
    puts "Usage: vitis_hls hls/run_hls.tcl <hls_csim|csynth|cosim> <medium|production> ?top?"
    exit 2
}

set script_argc [llength $argv]

if {$script_argc < 2 || $script_argc > 3} {
    usage
}

set action [lindex $argv 0]
set size_name [lindex $argv 1]
set top_name "wka_sar_top"
if {$script_argc == 3} {
    set top_name [lindex $argv 2]
}

array set size_values {
    medium 1
    production 2
}
if {![info exists size_values($size_name)]} {
    puts "ERROR: unsupported size profile '$size_name'"
    usage
}
if {$action ni {hls_csim csynth cosim}} {
    puts "ERROR: unsupported action '$action'"
    usage
}
if {$top_name ni {wka_sar_top corner_turn_top row_forward_top}} {
    puts "ERROR: unsupported top '$top_name'"
    usage
}
if {$action eq "hls_csim" && $top_name ne "wka_sar_top"} {
    puts "ERROR: C-sim through Vitis HLS uses the end-to-end wka_sar_top testbench"
    exit 2
}
if {$action eq "cosim" && $top_name ne "wka_sar_top"} {
    puts "ERROR: cosim uses the end-to-end wka_sar_top testbench"
    exit 2
}
if {$action eq "hls_csim" && $size_name ne "production"} {
    puts "ERROR: C-sim through Vitis HLS is fixed at the complete production raster"
    exit 2
}
if {$action eq "csynth" && $size_name ne "production"} {
    puts "ERROR: csynth is fixed at the complete production raster"
    exit 2
}
if {$action eq "cosim" && $size_name ne "medium"} {
    puts "ERROR: hand-written RTL cosim is fixed at 256x256"
    exit 2
}

source [file join $script_dir targets vu13p.tcl]

proc xml_value {xml tag} {
    set pattern [format {<%s>([^<]+)</%s>} $tag $tag]
    if {![regexp -- $pattern $xml _ value]} {
        error "synthesis report is missing $tag"
    }
    return $value
}

proc validate_csynth_report {report_path} {
    global WKA_CLOCK_PERIOD_NS WKA_CLOCK_UNCERTAINTY_NS
    global WKA_BRAM18K_BUDGET WKA_URAM_BUDGET WKA_DSP_BUDGET
    global WKA_FF_BUDGET WKA_LUT_BUDGET

    set handle [open $report_path r]
    set xml [read $handle]
    close $handle

    set estimated [expr {double([xml_value $xml EstimatedClockPeriod])}]
    set timing_budget [expr {$WKA_CLOCK_PERIOD_NS - $WKA_CLOCK_UNCERTAINTY_NS}]
    if {$estimated > $timing_budget} {
        puts [format "WKA WARNING: estimated clock %.3f ns misses the %.3f ns effective scheduling budget; synthesis continues" $estimated $timing_budget]
    }

    if {![regexp {(?s)<Resources>(.*?)</Resources>} $xml _ resources]} {
        error "synthesis report is missing resource estimates"
    }
    foreach {name limit} [list \
        BRAM_18K $WKA_BRAM18K_BUDGET \
        URAM $WKA_URAM_BUDGET \
        DSP $WKA_DSP_BUDGET \
        FF $WKA_FF_BUDGET \
        LUT $WKA_LUT_BUDGET] {
        set used [expr {int([xml_value $resources $name])}]
        if {$used > $limit} {
            error "$name usage $used exceeds budget $limit"
        }
    }
}

set common_cflags "-std=c++14 -DWKA_SIZE_PROFILE=$size_values($size_name)"
set project_dir [file join $root_dir work hls \
    "${top_name}_${action}_${size_name}"]
set solution_name "solution_${action}_${size_name}"

file mkdir [file dirname $project_dir]
cd [file dirname $project_dir]
open_project -reset [file tail $project_dir]
set_top $top_name

set design_sources {
    corner_turn.cpp
    fft_core.cpp
    stolt_interpolation.cpp
    wka_top.cpp
    hls_module_tops.cpp
}
foreach source_file $design_sources {
    add_files [file join $root_dir $source_file] -cflags $common_cflags
}
add_files -tb [file join $root_dir wka_tb.cpp] -cflags $common_cflags

open_solution -reset $solution_name -flow_target vivado
set_part $WKA_PART
create_clock -period $WKA_CLOCK_PERIOD_NS -name default
set_clock_uncertainty $WKA_CLOCK_UNCERTAINTY_NS
config_interface -m_axi_addr64=true -m_axi_alignment_byte_size=64 -m_axi_max_widen_bitwidth=512
config_dataflow -strict_mode=warning

set tb_args ""
if {$size_name eq "production"} {
    set input_path [file normalize [file join $root_dir .. .. data alos_raw_16384x16384.bin]]
    set output_path [file join $root_dir reports sar_wka_output.bmp]
    set tb_args "$input_path $output_path"
} else {
    set reduced_golden [file join $root_dir reports reference_cosim_256.bin]
    if {[file exists $reduced_golden]} {
        set tb_args $reduced_golden
    }
}

if {$action eq "hls_csim"} {
    csim_design -O -clean -argv $tb_args
} elseif {$action eq "csynth"} {
    csynth_design
    validate_csynth_report [file join $project_dir $solution_name syn report \
        "${top_name}_csynth.xml"]
} else {
    csynth_design
    validate_csynth_report [file join $project_dir $solution_name syn report \
        "${top_name}_csynth.xml"]
    cosim_design -O -rtl verilog -trace_level none \
        -disable_dependency_check -argv $tb_args
}

exit
