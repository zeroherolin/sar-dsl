set script_dir [file dirname [file normalize [info script]]]
set root_dir [file dirname $script_dir]

proc usage {} {
    puts "Usage: vitis_hls hls/run_hls.tcl <csim|csynth|cosim> <smoke|medium|production> ?top?"
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
    smoke 0
    medium 1
    production 2
}
if {![info exists size_values($size_name)]} {
    puts "ERROR: unsupported size profile '$size_name'"
    usage
}
if {$action ni {csim csynth cosim}} {
    puts "ERROR: unsupported action '$action'"
    usage
}
if {$top_name ni {wka_sar_top corner_turn_top row_transform_top}} {
    puts "ERROR: unsupported top '$top_name'"
    usage
}
if {$action eq "csim" && $top_name ne "wka_sar_top"} {
    puts "ERROR: csim uses the end-to-end wka_sar_top testbench"
    exit 2
}

source [file join $script_dir targets vu13p.tcl]

set common_cflags "-std=c++11 -DWKA_SIZE_PROFILE=$size_values($size_name)"
set project_dir [file join $root_dir work hls "${top_name}_${size_name}"]
set solution_name "solution_${size_name}"

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
    set input_path [file join $root_dir data alos_raw_16384x16384.bin]
    set output_path [file join $root_dir reports sar_wka_output.bmp]
    set tb_args "$input_path $output_path"
} elseif {$size_name eq "smoke"} {
    set reduced_golden [file join $root_dir reports reference_smoke_64.bin]
    if {[file exists $reduced_golden]} {
        set tb_args $reduced_golden
    }
}

if {$action eq "csim"} {
    csim_design -clean -argv $tb_args
} elseif {$action eq "csynth"} {
    csynth_design
} else {
    if {$size_name eq "production"} {
        puts "ERROR: production RTL cosim is intentionally disabled; use reduced-size cosim plus production csynth"
        exit 2
    }
    csynth_design
    cosim_design -rtl verilog -trace_level none -argv $tb_args
}

exit
