# ! latency_avg.1_vs_1.5x_7x
# 
# alibaba.per_60mins.write_p100.alibaba_51.3.idx_12
# alibaba.per_60mins.offset_p10.alibaba_06.3.idx_1
# 
# ! size_avg.1_vs_1.5x_vs_7x
# 
# alibaba.per_60mins.overall_p10.alibaba_56.3.idx_17
# alibaba.per_60mins.overall_p10.alibaba_56.3.idx_49
# alibaba.per_60mins.size_p100.alibaba_9025.1.idx_9

# ! single

# alibaba.per_60mins.overall_p10.alibaba_56.3.idx_18

# ! single-2

# alibaba.per_60mins.offset_p10.alibaba_06.3.idx_33

##############################################
# READ
##############################################

# ! read_size_avg.1_1.5_2_3_9
# 
# alibaba.per_60mins.overall_p10.alibaba_56.3.idx_18 # 1x
# alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_47 # 1.5x
# alibaba.per_60mins.size_p100.alibaba_9025.1.idx_27 # 2x
# alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_45 # 3x
# alibaba.per_60mins.offset_p10.alibaba_06.3.idx_33 # 9x

# ! read_size_avg

# 1:   alibaba.per_60mins.overall_p10.alibaba_56.3.idx_18 # 1x
# 1.5: alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_47 # 1.5x
# 2:   alibaba.per_60mins.size_p100.alibaba_9025.1.idx_27 # 2x
# 3:   alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_45 # 3x
# 4:   alibaba.per_60mins.overall_p10.alibaba_56.3.idx_38 # 4x
# 5:   alibaba.per_60mins.overall_p10.alibaba_56.3.idx_31 # 5x
# 6:   alibaba.per_60mins.overall_p10.alibaba_56.3.idx_39 # 6x
# 7:   alibaba.per_60mins.write_p100.alibaba_51.3.idx_10 # 7x
# 8:   alibaba.per_60mins.offset_p10.alibaba_06.3.idx_15 # 8x
# 9:   alibaba.per_60mins.offset_p10.alibaba_06.3.idx_33 # 9x
# 10:  alibaba.per_60mins.write_p100.alibaba_51.3.idx_37 # 10x

# ! read_latency_avg

# 1:    alibaba.per_60mins.iops_p100.alibaba_9087.1.idx_1
# 1.2:  alibaba.per_60mins.overall_p10.alibaba_56.3.idx_8
# 1.5:  alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_23
# 2:    alibaba.per_60mins.overall_p10.alibaba_56.3.idx_42
# 3:    alibaba.per_60mins.size_p100.alibaba_9025.1.idx_21
# 4:    alibaba.per_60mins.size_p100.alibaba_9025.1.idx_13


# ! read_iat_avg

# 1:    alibaba.per_60mins.size_p100.alibaba_9025.1.idx_7
# 1.2:  alibaba.per_60mins.overall_p10.alibaba_56.3.idx_10
# 1.5:  alibaba.per_60mins.size_p100.alibaba_9025.1.idx_10
# 2:    alibaba.per_60mins.size_p100.alibaba_9025.1.idx_5
# 3:    alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_40
# 6:    alibaba.per_60mins.write_p100.alibaba_51.3.idx_56
# 7:    alibaba.per_60mins.overall_p10.alibaba_56.3.idx_4
# 10:   alibaba.per_60mins.overall_p10.alibaba_56.3.idx_56


# ! read_throughput_avg

# 1:    alibaba.per_60mins.overall_p10.alibaba_56.3.idx_18
# 1.2:  alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_38
# 2:    alibaba.per_60mins.iops_p100.alibaba_9087.1.idx_56
# 3:    alibaba.per_60mins.iops_p100.alibaba_9087.1.idx_28
# 4:    alibaba.per_60mins.iops_p100.alibaba_9087.1.idx_21
# 5:    alibaba.per_60mins.iops_p100.alibaba_9087.1.idx_2
# 7:    alibaba.per_60mins.offset_p10.alibaba_06.3.idx_5
# 8:    alibaba.per_60mins.write_p100.alibaba_51.3.idx_27
# 9:    alibaba.per_60mins.offset_p10.alibaba_06.3.idx_18
# 10:   alibaba.per_60mins.offset_p10.alibaba_06.3.idx_15

# ##############################################
# # WRITE
# ##############################################

# ! write_size_avg

# 1:    alibaba.per_60mins.iops_p10.alibaba_39.2.idx_1
# 1.5:  alibaba.per_60mins.offset_p10.alibaba_06.3.idx_4
# 2:    alibaba.per_60mins.iops_p100.alibaba_9087.1.idx_49
# 3:    alibaba.per_60mins.iops_p100.alibaba_9087.1.idx_5
# 4:    alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_12
# 5:    alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_26
# 6:    alibaba.per_60mins.size_p100.alibaba_9025.1.idx_10

# ! write_latency_avg

# 1:    alibaba.per_60mins.iops_p10.alibaba_39.2.idx_1
# 1.2:  alibaba.per_60mins.iops_p100.alibaba_9087.1.idx_2
# 1.5:  alibaba.per_60mins.iops_p100.alibaba_9087.1.idx_24
# 2:    alibaba.per_60mins.offset_p10.alibaba_06.3.idx_23
# 3:    alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_23
# 4:    alibaba.per_60mins.offset_p10.alibaba_06.3.idx_15

# ! write_iat_avg

# 1:    alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_28
# 1.2:  alibaba.per_60mins.iops_p100.alibaba_9087.1.idx_41
# 1.5:  alibaba.per_60mins.iops_p10.alibaba_39.2.idx_1
# 2:    alibaba.per_60mins.overall_p10.alibaba_56.3.idx_28
# 3:    alibaba.per_60mins.overall_p10.alibaba_56.3.idx_11

# ! write_throughput_avg

# 1:    alibaba.per_60mins.offset_p10.alibaba_06.3.idx_15
# 1.2:  alibaba.per_60mins.offset_p10.alibaba_06.3.idx_34
# 1.5:  alibaba.per_60mins.overall_p10.alibaba_56.3.idx_9
# 2:    alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_37
# 3:    alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_26

# ##############################################
# # GENERAL
# ##############################################

# ! size_avg

# 1:    alibaba.per_60mins.overall_p10.alibaba_56.3.idx_17
# 1.2:  alibaba.per_60mins.offset_p10.alibaba_06.3.idx_30
# 1.5:  alibaba.per_60mins.iops_p100.alibaba_9087.1.idx_49
# 2:    alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_40
# 3:    alibaba.per_60mins.overall_p10.alibaba_56.3.idx_34
# 5:    alibaba.per_60mins.size_p100.alibaba_9025.1.idx_19
# 7:    alibaba.per_60mins.size_p100.alibaba_9025.1.idx_9

# ! latency_avg

# 1:    alibaba.per_60mins.write_p100.alibaba_51.3.idx_12
# 1.2:  alibaba.per_60mins.offset_p10.alibaba_06.3.idx_11
# 1.5:  alibaba.per_60mins.offset_p10.alibaba_06.3.idx_1
# 2:    alibaba.per_60mins.iops_p100.alibaba_9087.1.idx_47
# 3:    alibaba.per_60mins.size_p100.alibaba_9025.1.idx_2
# 6:    alibaba.per_60mins.size_p100.alibaba_9025.1.idx_13

# ! iat_avg

# 1:    alibaba.per_60mins.iops_p100.alibaba_9087.1.idx_11
# 1.2:  alibaba.per_60mins.iops_p100.alibaba_9087.1.idx_1
# 1.5:  alibaba.per_60mins.iops_p100.alibaba_9087.1.idx_43
# 2:    alibaba.per_60mins.offset_p10.alibaba_06.3.idx_2
# 3:    alibaba.per_60mins.write_p100.alibaba_51.3.idx_3

# ! throughput_avg

# 1:    alibaba.per_60mins.overall_p10.alibaba_56.3.idx_10
# 1.2:  alibaba.per_60mins.overall_p10.alibaba_56.3.idx_2
# 1.5:  alibaba.per_60mins.offset_p10.alibaba_06.3.idx_4
# 2:    alibaba.per_60mins.iops_p100.alibaba_9087.1.idx_9
# 3:    alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_1


# ! same_read_size_3x

# alibaba.per_60mins.iops_p100.alibaba_9087.1.idx_37
# alibaba.per_60mins.iops_p100.alibaba_9087.1.idx_38
# alibaba.per_60mins.iops_p100.alibaba_9087.1.idx_39
# alibaba.per_60mins.iops_p100.alibaba_9087.1.idx_46
# alibaba.per_60mins.iops_p100.alibaba_9087.1.idx_52
# alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_25
# alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_45
# alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_52
# alibaba.per_60mins.write_p100.alibaba_51.3.idx_16

# ! same_read_throughput_3x

# alibaba.per_60mins.iops_p100.alibaba_9087.1.idx_28
# alibaba.per_60mins.iops_p100.alibaba_9087.1.idx_49
# alibaba.per_60mins.iops_p100.alibaba_9087.1.idx_50
# alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_17
# alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_20
# alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_25
# alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_56
# alibaba.per_60mins.overall_p10.alibaba_56.3.idx_15
# alibaba.per_60mins.overall_p10.alibaba_56.3.idx_29
# alibaba.per_60mins.overall_p10.alibaba_56.3.idx_32
# alibaba.per_60mins.overall_p10.alibaba_56.3.idx_35
# alibaba.per_60mins.overall_p10.alibaba_56.3.idx_39
# alibaba.per_60mins.size_p100.alibaba_9025.1.idx_7
# alibaba.per_60mins.write_p100.alibaba_51.3.idx_33

# ! same_latency_2x

# alibaba.per_60mins.iops_p100.alibaba_9087.1.idx_11
# alibaba.per_60mins.iops_p100.alibaba_9087.1.idx_47
# alibaba.per_60mins.iops_p100.alibaba_9087.1.idx_48
# alibaba.per_60mins.iops_p100.alibaba_9087.1.idx_50
# alibaba.per_60mins.iops_p100.alibaba_9087.1.idx_51
# alibaba.per_60mins.iops_p100.alibaba_9087.1.idx_52
# alibaba.per_60mins.offset_p10.alibaba_06.3.idx_5
# alibaba.per_60mins.offset_p10.alibaba_06.3.idx_9
# alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_8
# alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_18
# alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_34
# alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_37
# alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_38
# alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_40

# ! same_read_latency_3x

# alibaba.per_60mins.offset_p10.alibaba_06.3.idx_15
# alibaba.per_60mins.overall_p10.alibaba_56.3.idx_57
# alibaba.per_60mins.size_p100.alibaba_9025.1.idx_19
# alibaba.per_60mins.size_p100.alibaba_9025.1.idx_20
# alibaba.per_60mins.size_p100.alibaba_9025.1.idx_21
# alibaba.per_60mins.write_p100.alibaba_51.3.idx_26

# ! same_throughput_1.2x

# alibaba.per_60mins.overall_p10.alibaba_56.3.idx_2
# alibaba.per_60mins.overall_p10.alibaba_56.3.idx_6
# alibaba.per_60mins.overall_p10.alibaba_56.3.idx_7
# alibaba.per_60mins.overall_p10.alibaba_56.3.idx_8
# alibaba.per_60mins.overall_p10.alibaba_56.3.idx_9
# alibaba.per_60mins.overall_p10.alibaba_56.3.idx_11
# alibaba.per_60mins.overall_p10.alibaba_56.3.idx_12
# alibaba.per_60mins.overall_p10.alibaba_56.3.idx_13
# alibaba.per_60mins.overall_p10.alibaba_56.3.idx_14
# alibaba.per_60mins.overall_p10.alibaba_56.3.idx_17
# alibaba.per_60mins.overall_p10.alibaba_56.3.idx_18
# alibaba.per_60mins.overall_p10.alibaba_56.3.idx_23

# ! same_throughput_3x

# alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_1
# alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_5
# alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_13
# alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_16
# alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_23
# alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_27
# alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_31
# alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_45
# alibaba.per_60mins.size_p100.alibaba_9025.1.idx_9

# ! same_throughput_2x

# alibaba.per_60mins.iops_p100.alibaba_9087.1.idx_9
# alibaba.per_60mins.iops_p100.alibaba_9087.1.idx_24
# alibaba.per_60mins.offset_p10.alibaba_06.3.idx_19
# alibaba.per_60mins.offset_p10.alibaba_06.3.idx_36
# alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_40
# alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_53
# alibaba.per_60mins.size_p100.alibaba_9025.1.idx_28
# alibaba.per_60mins.size_p100.alibaba_9025.1.idx_44
# alibaba.per_60mins.write_p100.alibaba_51.3.idx_3
# alibaba.per_60mins.write_p100.alibaba_51.3.idx_10


# ! same_size_2x

# alibaba.per_60mins.iops_p100.alibaba_9087.1.idx_3
# alibaba.per_60mins.iops_p100.alibaba_9087.1.idx_7
# alibaba.per_60mins.iops_p100.alibaba_9087.1.idx_17
# alibaba.per_60mins.iops_p100.alibaba_9087.1.idx_28
# alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_41
# alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_49
# alibaba.per_60mins.overall_p100.alibaba_9030.0.idx_55
# alibaba.per_60mins.overall_p10.alibaba_56.3.idx_26
# alibaba.per_60mins.overall_p10.alibaba_56.3.idx_56
# alibaba.per_60mins.size_p100.alibaba_9025.1.idx_30
# alibaba.per_60mins.size_p100.alibaba_9025.1.idx_35
# alibaba.per_60mins.size_p100.alibaba_9025.1.idx_43

! size_avg_single_file

1:    alibaba.per_60mins.overall_p10.alibaba_56.3.idx_17
1.2:  alibaba.per_60mins.overall_p10.alibaba_56.3.idx_37
1.5:  alibaba.per_60mins.overall_p10.alibaba_56.3.idx_49
2:    alibaba.per_60mins.overall_p10.alibaba_56.3.idx_56
3:    alibaba.per_60mins.overall_p10.alibaba_56.3.idx_34