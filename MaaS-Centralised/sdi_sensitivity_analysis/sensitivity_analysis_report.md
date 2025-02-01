# sdi Sensitivity Analysis Report

## FPS Analysis Results

### Data Summary
- Number of samples: 72
- Parameters analyzed: simulation_id, timestamp, analysis_type, utility_coefficients_beta_C, utility_coefficients_beta_T, utility_coefficients_beta_W, utility_coefficients_beta_A, value_of_time_low, value_of_time_middle, value_of_time_high, uber_parameters_uber_like1_capacity, uber_parameters_uber_like1_price, uber_parameters_uber_like2_capacity, uber_parameters_uber_like2_price, bike_parameters_bike_share1_capacity, bike_parameters_bike_share1_price, bike_parameters_bike_share2_capacity, bike_parameters_bike_share2_price, maas_surcharge_S_base, maas_surcharge_alpha, maas_surcharge_delta, public_transport_train_on_peak, public_transport_train_off_peak, public_transport_bus_on_peak, public_transport_bus_off_peak, congestion_params_alpha, congestion_params_beta, congestion_params_capacity, subsidy_type, subsidy_percentages_low_bike, subsidy_percentages_low_car, subsidy_percentages_low_MaaS_Bundle, subsidy_percentages_middle_bike, subsidy_percentages_middle_car, subsidy_percentages_middle_MaaS_Bundle, subsidy_percentages_high_bike, subsidy_percentages_high_car, subsidy_percentages_high_MaaS_Bundle, varied_mode
- Metrics analyzed: sdi, sur, mae, upi

### Parameter Impacts

#### simulation_id
- Mean value: 35.500
- Standard deviation: 20.928
- Correlations with metrics:
  - sdi: r=0.074 (p=0.535)
  - sur: r=0.067 (p=0.579)
  - mae: r=0.109 (p=0.363)
  - upi: r=0.001 (p=0.997)

#### utility_coefficients_beta_C
- Mean value: -0.050
- Standard deviation: 0.017
- Correlations with metrics:
  - sdi: r=-0.182 (p=0.127)
  - sur: r=-0.053 (p=0.656)
  - mae: r=-0.169 (p=0.156)
  - upi: r=-0.178 (p=0.134)

#### utility_coefficients_beta_T
- Mean value: -0.049
- Standard deviation: 0.017
- Correlations with metrics:
  - sdi: r=-0.073 (p=0.540)
  - sur: r=-0.068 (p=0.571)
  - mae: r=-0.061 (p=0.610)
  - upi: r=-0.072 (p=0.550)

#### utility_coefficients_beta_W
- Mean value: -0.009
- Standard deviation: 0.005
- Correlations with metrics:
  - sdi: r=0.068 (p=0.569)
  - sur: r=-0.025 (p=0.835)
  - mae: r=0.061 (p=0.614)
  - upi: r=0.082 (p=0.491)

#### utility_coefficients_beta_A
- Mean value: -0.009
- Standard deviation: 0.006
- Correlations with metrics:
  - sdi: r=0.100 (p=0.403)
  - sur: r=0.029 (p=0.812)
  - mae: r=0.097 (p=0.417)
  - upi: r=0.092 (p=0.442)

#### value_of_time_low
- Mean value: 9.640
- Standard deviation: 0.000
- Correlations with metrics:
  - sdi: r=nan (p=nan)
  - sur: r=nan (p=nan)
  - mae: r=nan (p=nan)
  - upi: r=nan (p=nan)

#### value_of_time_middle
- Mean value: 23.700
- Standard deviation: 0.000
- Correlations with metrics:
  - sdi: r=nan (p=nan)
  - sur: r=nan (p=nan)
  - mae: r=nan (p=nan)
  - upi: r=nan (p=nan)

#### value_of_time_high
- Mean value: 67.200
- Standard deviation: 0.000
- Correlations with metrics:
  - sdi: r=nan (p=nan)
  - sur: r=nan (p=nan)
  - mae: r=nan (p=nan)
  - upi: r=nan (p=nan)

#### uber_parameters_uber_like1_capacity
- Mean value: 7.528
- Standard deviation: 1.074
- Correlations with metrics:
  - sdi: r=0.247 (p=0.036)
  - sur: r=0.386 (p=0.001)
  - mae: r=0.191 (p=0.108)
  - upi: r=0.227 (p=0.055)

#### uber_parameters_uber_like1_price
- Mean value: 6.103
- Standard deviation: 1.056
- Correlations with metrics:
  - sdi: r=-0.062 (p=0.604)
  - sur: r=-0.175 (p=0.142)
  - mae: r=-0.040 (p=0.738)
  - upi: r=-0.051 (p=0.673)

#### uber_parameters_uber_like2_capacity
- Mean value: 8.375
- Standard deviation: 1.119
- Correlations with metrics:
  - sdi: r=-0.031 (p=0.798)
  - sur: r=-0.005 (p=0.966)
  - mae: r=-0.011 (p=0.927)
  - upi: r=-0.058 (p=0.627)

#### uber_parameters_uber_like2_price
- Mean value: 6.523
- Standard deviation: 1.108
- Correlations with metrics:
  - sdi: r=-0.035 (p=0.769)
  - sur: r=-0.087 (p=0.467)
  - mae: r=-0.036 (p=0.762)
  - upi: r=-0.010 (p=0.931)

#### bike_parameters_bike_share1_capacity
- Mean value: 9.347
- Standard deviation: 1.153
- Correlations with metrics:
  - sdi: r=0.093 (p=0.436)
  - sur: r=0.031 (p=0.797)
  - mae: r=0.019 (p=0.877)
  - upi: r=0.197 (p=0.098)

#### bike_parameters_bike_share1_price
- Mean value: 0.995
- Standard deviation: 0.105
- Correlations with metrics:
  - sdi: r=0.042 (p=0.726)
  - sur: r=-0.270 (p=0.022)
  - mae: r=0.116 (p=0.334)
  - upi: r=-0.009 (p=0.940)

#### bike_parameters_bike_share2_capacity
- Mean value: 11.569
- Standard deviation: 1.136
- Correlations with metrics:
  - sdi: r=-0.196 (p=0.099)
  - sur: r=-0.147 (p=0.219)
  - mae: r=-0.167 (p=0.161)
  - upi: r=-0.194 (p=0.102)

#### bike_parameters_bike_share2_price
- Mean value: 1.232
- Standard deviation: 0.112
- Correlations with metrics:
  - sdi: r=-0.153 (p=0.200)
  - sur: r=-0.113 (p=0.345)
  - mae: r=-0.076 (p=0.523)
  - upi: r=-0.236 (p=0.046)

#### maas_surcharge_S_base
- Mean value: 0.084
- Standard deviation: 0.000
- Correlations with metrics:
  - sdi: r=nan (p=nan)
  - sur: r=nan (p=nan)
  - mae: r=nan (p=nan)
  - upi: r=nan (p=nan)

#### maas_surcharge_alpha
- Mean value: 0.214
- Standard deviation: 0.000
- Correlations with metrics:
  - sdi: r=nan (p=nan)
  - sur: r=nan (p=nan)
  - mae: r=nan (p=nan)
  - upi: r=nan (p=nan)

#### maas_surcharge_delta
- Mean value: 0.510
- Standard deviation: 0.000
- Correlations with metrics:
  - sdi: r=nan (p=nan)
  - sur: r=nan (p=nan)
  - mae: r=nan (p=nan)
  - upi: r=nan (p=nan)

#### public_transport_train_on_peak
- Mean value: 2.041
- Standard deviation: 0.282
- Correlations with metrics:
  - sdi: r=0.047 (p=0.697)
  - sur: r=0.066 (p=0.581)
  - mae: r=0.045 (p=0.705)
  - upi: r=0.030 (p=0.803)

#### public_transport_train_off_peak
- Mean value: 1.454
- Standard deviation: 0.297
- Correlations with metrics:
  - sdi: r=-0.076 (p=0.528)
  - sur: r=-0.142 (p=0.235)
  - mae: r=-0.034 (p=0.774)
  - upi: r=-0.101 (p=0.400)

#### public_transport_bus_on_peak
- Mean value: 0.977
- Standard deviation: 0.113
- Correlations with metrics:
  - sdi: r=0.162 (p=0.174)
  - sur: r=0.012 (p=0.920)
  - mae: r=0.168 (p=0.159)
  - upi: r=0.141 (p=0.236)

#### public_transport_bus_off_peak
- Mean value: 0.803
- Standard deviation: 0.109
- Correlations with metrics:
  - sdi: r=0.073 (p=0.541)
  - sur: r=0.036 (p=0.765)
  - mae: r=0.039 (p=0.742)
  - upi: r=0.113 (p=0.345)

#### congestion_params_alpha
- Mean value: 0.250
- Standard deviation: 0.028
- Correlations with metrics:
  - sdi: r=0.009 (p=0.942)
  - sur: r=0.027 (p=0.825)
  - mae: r=-0.040 (p=0.740)
  - upi: r=0.077 (p=0.519)

#### congestion_params_beta
- Mean value: 3.980
- Standard deviation: 0.558
- Correlations with metrics:
  - sdi: r=0.200 (p=0.092)
  - sur: r=0.173 (p=0.146)
  - mae: r=0.152 (p=0.204)
  - upi: r=0.222 (p=0.061)

#### congestion_params_capacity
- Mean value: 3.942
- Standard deviation: 0.603
- Correlations with metrics:
  - sdi: r=0.003 (p=0.980)
  - sur: r=-0.059 (p=0.621)
  - mae: r=-0.000 (p=0.997)
  - upi: r=0.022 (p=0.853)

#### subsidy_percentages_low_bike
- Mean value: 0.297
- Standard deviation: 0.035
- Correlations with metrics:
  - sdi: r=-0.025 (p=0.832)
  - sur: r=-0.017 (p=0.884)
  - mae: r=0.005 (p=0.970)
  - upi: r=-0.066 (p=0.580)

#### subsidy_percentages_low_car
- Mean value: 0.200
- Standard deviation: 0.017
- Correlations with metrics:
  - sdi: r=0.100 (p=0.405)
  - sur: r=0.079 (p=0.509)
  - mae: r=0.051 (p=0.671)
  - upi: r=0.150 (p=0.207)

#### subsidy_percentages_low_MaaS_Bundle
- Mean value: 0.454
- Standard deviation: 0.052
- Correlations with metrics:
  - sdi: r=0.020 (p=0.867)
  - sur: r=0.053 (p=0.657)
  - mae: r=0.032 (p=0.788)
  - upi: r=-0.013 (p=0.916)

#### subsidy_percentages_middle_bike
- Mean value: 0.247
- Standard deviation: 0.035
- Correlations with metrics:
  - sdi: r=-0.025 (p=0.832)
  - sur: r=-0.017 (p=0.884)
  - mae: r=0.005 (p=0.970)
  - upi: r=-0.066 (p=0.580)

#### subsidy_percentages_middle_car
- Mean value: 0.150
- Standard deviation: 0.017
- Correlations with metrics:
  - sdi: r=0.100 (p=0.405)
  - sur: r=0.079 (p=0.509)
  - mae: r=0.051 (p=0.671)
  - upi: r=0.150 (p=0.207)

#### subsidy_percentages_middle_MaaS_Bundle
- Mean value: 0.379
- Standard deviation: 0.044
- Correlations with metrics:
  - sdi: r=0.020 (p=0.867)
  - sur: r=0.053 (p=0.657)
  - mae: r=0.032 (p=0.788)
  - upi: r=-0.013 (p=0.916)

#### subsidy_percentages_high_bike
- Mean value: 0.197
- Standard deviation: 0.035
- Correlations with metrics:
  - sdi: r=-0.025 (p=0.832)
  - sur: r=-0.017 (p=0.884)
  - mae: r=0.005 (p=0.970)
  - upi: r=-0.066 (p=0.580)

#### subsidy_percentages_high_car
- Mean value: 0.100
- Standard deviation: 0.017
- Correlations with metrics:
  - sdi: r=0.100 (p=0.405)
  - sur: r=0.079 (p=0.509)
  - mae: r=0.051 (p=0.671)
  - upi: r=0.150 (p=0.207)

#### subsidy_percentages_high_MaaS_Bundle
- Mean value: 0.303
- Standard deviation: 0.035
- Correlations with metrics:
  - sdi: r=0.020 (p=0.867)
  - sur: r=0.053 (p=0.657)
  - mae: r=0.032 (p=0.788)
  - upi: r=-0.013 (p=0.916)

### Parameter Interactions
See interaction heatmap visualization for details.

### Temporal Effects
See temporal sensitivity plots for details.

---

## PBS Analysis Results

### Data Summary
- Number of samples: 216
- Parameters analyzed: simulation_id, timestamp, analysis_type, utility_coefficients_beta_C, utility_coefficients_beta_T, utility_coefficients_beta_W, utility_coefficients_beta_A, value_of_time_low, value_of_time_middle, value_of_time_high, uber_parameters_uber_like1_capacity, uber_parameters_uber_like1_price, uber_parameters_uber_like2_capacity, uber_parameters_uber_like2_price, bike_parameters_bike_share1_capacity, bike_parameters_bike_share1_price, bike_parameters_bike_share2_capacity, bike_parameters_bike_share2_price, maas_surcharge_S_base, maas_surcharge_alpha, maas_surcharge_delta, public_transport_train_on_peak, public_transport_train_off_peak, public_transport_bus_on_peak, public_transport_bus_off_peak, congestion_params_alpha, congestion_params_beta, congestion_params_capacity, subsidy_type, subsidy_percentages_low_bike, subsidy_percentages_low_car, subsidy_percentages_low_MaaS_Bundle, subsidy_percentages_middle_bike, subsidy_percentages_middle_car, subsidy_percentages_middle_MaaS_Bundle, subsidy_percentages_high_bike, subsidy_percentages_high_car, subsidy_percentages_high_MaaS_Bundle, varied_mode
- Metrics analyzed: sdi, sur, mae, upi

### Parameter Impacts

#### simulation_id
- Mean value: 35.500
- Standard deviation: 20.831
- Correlations with metrics:
  - sdi: r=-0.048 (p=0.482)
  - sur: r=0.230 (p=0.001)
  - mae: r=-0.071 (p=0.298)
  - upi: r=-0.017 (p=0.803)

#### utility_coefficients_beta_C
- Mean value: -0.050
- Standard deviation: 0.017
- Correlations with metrics:
  - sdi: r=-0.131 (p=0.054)
  - sur: r=-0.096 (p=0.159)
  - mae: r=-0.110 (p=0.107)
  - upi: r=-0.136 (p=0.046)

#### utility_coefficients_beta_T
- Mean value: -0.049
- Standard deviation: 0.017
- Correlations with metrics:
  - sdi: r=-0.110 (p=0.106)
  - sur: r=-0.052 (p=0.447)
  - mae: r=-0.093 (p=0.176)
  - upi: r=-0.120 (p=0.077)

#### utility_coefficients_beta_W
- Mean value: -0.009
- Standard deviation: 0.005
- Correlations with metrics:
  - sdi: r=-0.015 (p=0.822)
  - sur: r=0.031 (p=0.650)
  - mae: r=-0.000 (p=0.998)
  - upi: r=-0.059 (p=0.390)

#### utility_coefficients_beta_A
- Mean value: -0.009
- Standard deviation: 0.006
- Correlations with metrics:
  - sdi: r=0.028 (p=0.686)
  - sur: r=0.023 (p=0.733)
  - mae: r=0.050 (p=0.461)
  - upi: r=-0.046 (p=0.506)

#### value_of_time_low
- Mean value: 9.640
- Standard deviation: 0.000
- Correlations with metrics:
  - sdi: r=nan (p=nan)
  - sur: r=nan (p=nan)
  - mae: r=nan (p=nan)
  - upi: r=nan (p=nan)

#### value_of_time_middle
- Mean value: 23.700
- Standard deviation: 0.000
- Correlations with metrics:
  - sdi: r=nan (p=nan)
  - sur: r=nan (p=nan)
  - mae: r=nan (p=nan)
  - upi: r=nan (p=nan)

#### value_of_time_high
- Mean value: 67.200
- Standard deviation: 0.000
- Correlations with metrics:
  - sdi: r=nan (p=nan)
  - sur: r=nan (p=nan)
  - mae: r=nan (p=nan)
  - upi: r=nan (p=nan)

#### uber_parameters_uber_like1_capacity
- Mean value: 7.528
- Standard deviation: 1.069
- Correlations with metrics:
  - sdi: r=0.181 (p=0.008)
  - sur: r=0.177 (p=0.009)
  - mae: r=0.149 (p=0.028)
  - upi: r=0.185 (p=0.006)

#### uber_parameters_uber_like1_price
- Mean value: 6.103
- Standard deviation: 1.051
- Correlations with metrics:
  - sdi: r=-0.023 (p=0.733)
  - sur: r=-0.132 (p=0.052)
  - mae: r=0.007 (p=0.913)
  - upi: r=-0.076 (p=0.269)

#### uber_parameters_uber_like2_capacity
- Mean value: 8.375
- Standard deviation: 1.114
- Correlations with metrics:
  - sdi: r=0.009 (p=0.894)
  - sur: r=0.101 (p=0.141)
  - mae: r=0.022 (p=0.745)
  - upi: r=-0.048 (p=0.485)

#### uber_parameters_uber_like2_price
- Mean value: 6.523
- Standard deviation: 1.103
- Correlations with metrics:
  - sdi: r=-0.071 (p=0.299)
  - sur: r=-0.123 (p=0.071)
  - mae: r=-0.085 (p=0.212)
  - upi: r=0.010 (p=0.889)

#### bike_parameters_bike_share1_capacity
- Mean value: 9.347
- Standard deviation: 1.147
- Correlations with metrics:
  - sdi: r=0.101 (p=0.139)
  - sur: r=0.069 (p=0.315)
  - mae: r=0.103 (p=0.133)
  - upi: r=0.058 (p=0.395)

#### bike_parameters_bike_share1_price
- Mean value: 0.995
- Standard deviation: 0.105
- Correlations with metrics:
  - sdi: r=0.141 (p=0.039)
  - sur: r=-0.051 (p=0.452)
  - mae: r=0.190 (p=0.005)
  - upi: r=-0.020 (p=0.774)

#### bike_parameters_bike_share2_capacity
- Mean value: 11.569
- Standard deviation: 1.131
- Correlations with metrics:
  - sdi: r=-0.128 (p=0.060)
  - sur: r=-0.048 (p=0.478)
  - mae: r=-0.105 (p=0.124)
  - upi: r=-0.149 (p=0.029)

#### bike_parameters_bike_share2_price
- Mean value: 1.232
- Standard deviation: 0.111
- Correlations with metrics:
  - sdi: r=-0.026 (p=0.708)
  - sur: r=-0.044 (p=0.518)
  - mae: r=0.029 (p=0.672)
  - upi: r=-0.159 (p=0.020)

#### maas_surcharge_S_base
- Mean value: 0.084
- Standard deviation: 0.000
- Correlations with metrics:
  - sdi: r=nan (p=nan)
  - sur: r=nan (p=nan)
  - mae: r=nan (p=nan)
  - upi: r=nan (p=nan)

#### maas_surcharge_alpha
- Mean value: 0.214
- Standard deviation: 0.000
- Correlations with metrics:
  - sdi: r=nan (p=nan)
  - sur: r=nan (p=nan)
  - mae: r=nan (p=nan)
  - upi: r=nan (p=nan)

#### maas_surcharge_delta
- Mean value: 0.510
- Standard deviation: 0.000
- Correlations with metrics:
  - sdi: r=nan (p=nan)
  - sur: r=nan (p=nan)
  - mae: r=nan (p=nan)
  - upi: r=nan (p=nan)

#### public_transport_train_on_peak
- Mean value: 2.041
- Standard deviation: 0.280
- Correlations with metrics:
  - sdi: r=0.037 (p=0.592)
  - sur: r=0.004 (p=0.954)
  - mae: r=0.024 (p=0.727)
  - upi: r=0.061 (p=0.371)

#### public_transport_train_off_peak
- Mean value: 1.454
- Standard deviation: 0.295
- Correlations with metrics:
  - sdi: r=-0.146 (p=0.032)
  - sur: r=0.141 (p=0.038)
  - mae: r=-0.153 (p=0.024)
  - upi: r=-0.117 (p=0.086)

#### public_transport_bus_on_peak
- Mean value: 0.977
- Standard deviation: 0.112
- Correlations with metrics:
  - sdi: r=0.130 (p=0.057)
  - sur: r=-0.007 (p=0.914)
  - mae: r=0.140 (p=0.040)
  - upi: r=0.071 (p=0.300)

#### public_transport_bus_off_peak
- Mean value: 0.803
- Standard deviation: 0.109
- Correlations with metrics:
  - sdi: r=0.022 (p=0.753)
  - sur: r=0.024 (p=0.730)
  - mae: r=0.027 (p=0.693)
  - upi: r=-0.004 (p=0.959)

#### congestion_params_alpha
- Mean value: 0.250
- Standard deviation: 0.028
- Correlations with metrics:
  - sdi: r=-0.081 (p=0.234)
  - sur: r=-0.110 (p=0.106)
  - mae: r=-0.093 (p=0.173)
  - upi: r=-0.008 (p=0.911)

#### congestion_params_beta
- Mean value: 3.980
- Standard deviation: 0.556
- Correlations with metrics:
  - sdi: r=0.078 (p=0.255)
  - sur: r=0.086 (p=0.210)
  - mae: r=0.053 (p=0.434)
  - upi: r=0.107 (p=0.116)

#### congestion_params_capacity
- Mean value: 3.942
- Standard deviation: 0.600
- Correlations with metrics:
  - sdi: r=0.083 (p=0.226)
  - sur: r=0.050 (p=0.467)
  - mae: r=0.076 (p=0.267)
  - upi: r=0.071 (p=0.301)

#### subsidy_percentages_low_bike
- Mean value: 0.297
- Standard deviation: 0.035
- Correlations with metrics:
  - sdi: r=-0.046 (p=0.503)
  - sur: r=0.042 (p=0.543)
  - mae: r=-0.041 (p=0.548)
  - upi: r=-0.055 (p=0.423)

#### subsidy_percentages_low_car
- Mean value: 0.200
- Standard deviation: 0.017
- Correlations with metrics:
  - sdi: r=-0.042 (p=0.538)
  - sur: r=0.061 (p=0.370)
  - mae: r=-0.052 (p=0.448)
  - upi: r=-0.017 (p=0.806)

#### subsidy_percentages_low_MaaS_Bundle
- Mean value: 0.454
- Standard deviation: 0.052
- Correlations with metrics:
  - sdi: r=-0.128 (p=0.061)
  - sur: r=-0.033 (p=0.629)
  - mae: r=-0.146 (p=0.033)
  - upi: r=-0.041 (p=0.551)

#### subsidy_percentages_middle_bike
- Mean value: 0.247
- Standard deviation: 0.035
- Correlations with metrics:
  - sdi: r=-0.046 (p=0.503)
  - sur: r=0.042 (p=0.543)
  - mae: r=-0.041 (p=0.548)
  - upi: r=-0.055 (p=0.423)

#### subsidy_percentages_middle_car
- Mean value: 0.150
- Standard deviation: 0.017
- Correlations with metrics:
  - sdi: r=-0.042 (p=0.538)
  - sur: r=0.061 (p=0.370)
  - mae: r=-0.052 (p=0.448)
  - upi: r=-0.017 (p=0.806)

#### subsidy_percentages_middle_MaaS_Bundle
- Mean value: 0.379
- Standard deviation: 0.044
- Correlations with metrics:
  - sdi: r=-0.128 (p=0.061)
  - sur: r=-0.033 (p=0.629)
  - mae: r=-0.146 (p=0.033)
  - upi: r=-0.041 (p=0.551)

#### subsidy_percentages_high_bike
- Mean value: 0.197
- Standard deviation: 0.035
- Correlations with metrics:
  - sdi: r=-0.046 (p=0.503)
  - sur: r=0.042 (p=0.543)
  - mae: r=-0.041 (p=0.548)
  - upi: r=-0.055 (p=0.423)

#### subsidy_percentages_high_car
- Mean value: 0.100
- Standard deviation: 0.017
- Correlations with metrics:
  - sdi: r=-0.042 (p=0.538)
  - sur: r=0.061 (p=0.370)
  - mae: r=-0.052 (p=0.448)
  - upi: r=-0.017 (p=0.806)

#### subsidy_percentages_high_MaaS_Bundle
- Mean value: 0.303
- Standard deviation: 0.035
- Correlations with metrics:
  - sdi: r=-0.128 (p=0.061)
  - sur: r=-0.033 (p=0.629)
  - mae: r=-0.146 (p=0.033)
  - upi: r=-0.041 (p=0.551)

### Parameter Interactions
See interaction heatmap visualization for details.

### Temporal Effects
See temporal sensitivity plots for details.

---

