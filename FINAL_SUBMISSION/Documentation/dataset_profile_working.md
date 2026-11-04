# Factual Working Profile: Chiller Operational Dataset

**Document Purpose:** Factual record of raw dataset properties discovered during Phase 1 inspection. Unpolished engineering log; contains observed properties without synthetic assumptions or ghostwritten narrative.

---

## 1. File and Format Verification
- **File Name:** `Analyse Chillers_AI Course.xls`
- **Location in Repository:** `data/raw/official_dataset/Analyse Chillers_AI Course.xls`
- **Root Directory Copy:** `Analyse Chillers_AI Course.xls` (verified identical, not modified/deleted)
- **File Size:** 665,088 bytes (649.5 KB)
- **SHA-256 Hash:** `7c3e80d5fdf3b97a4b833a6b0cd23fd81ba9b9ad860e6af42e19976a414c4914` (both copies match 100%)
- **Container Format:** Microsoft Excel 97-2004 Workbook (BIFF8 / OLE Compound File)
- **Total Worksheets in Workbook:** 1 worksheet discovered (`wb.nsheets == 1`).
  - Sheet 1: `Log_Chiller_South` (Visibility: Visible, Dimensions: 525 rows x 60 columns).
  - Hidden / Very Hidden Sheets: None detected.
- **Header Offset Structure:**
  - Row 0 (Excel Row 1): Title string (`"Log - Chiller Plant"`), remainder empty/NaN.
  - Row 1 (Excel Row 2): Empty / all NaN cells.
  - Row 2 (Excel Row 3): Sensor variable headers (`Sl.No`, `CHILLER TAG`, `Date`, `Time`, ...).
  - Row 3 to 524 (Excel Rows 4 to 525): 522 observational data rows.
- **Reading Configuration:** `pd.read_excel(filepath, sheet_name='Log_Chiller_South', header=2)`

---

## 2. Dataset Dimensions and Time Coverage
- **Total Observational Rows:** 522 observations
- **Total Variables:** 60 columns
- **Date Range:** 2026-03-11 to 2026-03-20 (10 calendar days)
- **Time Coverage:** Hourly logs from `00:00:00` to `23:00:00` (24 distinct hours across days)
- **Total Unique Timestamps across Plant:** 222 hours
- **Raw Ordering:** The raw Excel file is logged in **reverse chronological order** (Row 1 = 2026-03-20 08:00; Row 522 = 2026-03-11 00:00).
- **Serial Number (`Sl.No`):** Monotonically increasing from 1 to 522 in raw file order (inverses true time).

---

## 3. Chiller Tag Distribution & Plant Architecture
The plant contains 5 separate chiller units logged in this log sheet:

| Chiller Tag | Observation Count | Start Timestamp (Earliest) | End Timestamp (Latest) | Sampling Interval | Duplicate Timestamps |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Chiller - 1** | 96 | 2026-03-11 00:00:00 | 2026-03-20 08:00:00 | Hourly (91 1h diffs, 1 15h, 1 4d17h gap) | 0 |
| **Chiller - 2** | 86 | 2026-03-11 10:00:00 | 2026-03-15 18:00:00 | Hourly (83 1h diffs, 1 4h, 1 17h gap) | 0 |
| **Chiller - 3** | 120 | 2026-03-12 08:00:00 | 2026-03-19 18:00:00 | Hourly (114 1h diffs, 1 3h, 1 22h gap) | 0 |
| **Chiller - 4** | 131 | 2026-03-11 09:00:00 | 2026-03-19 17:00:00 | Hourly (123 1h diffs, 2 3h, 2 5h gaps) | 0 |
| **Chiller - 5** | 89 | 2026-03-11 03:00:00 | 2026-03-20 08:00:00 | Hourly (76 1h diffs, 5 3h, 1 16h gap) | 0 |
| **Total** | **522** | **2026-03-11 00:00:00** | **2026-03-20 08:00:00** | - | **0 per chiller** |

- **Concurrent Chillers in Operation:**
  - 1 Chiller running: 22 timestamps
  - 2 Chillers running concurrently: 125 timestamps
  - 3 Chillers running concurrently: 50 timestamps
  - 4 Chillers running concurrently: 25 timestamps

---

## 4. Complete Variable Inventory & Domain Categorization

All 60 columns in the raw dataset categorized by sensor type and engineering subsystem:

### A. Identification & Timestamps (4 columns)
1. `Sl.No` (Integer, 1 to 522): Log entry serial number.
2. `CHILLER TAG` (String / Object): Unit identifier (`Chiller - 1` through `Chiller - 5`).
3. `Date` (datetime64): Calendar date (2026-03-11 to 2026-03-20).
4. `Time` (object / datetime.time): Time of logging (00:00 to 23:00).

### B. Core Temperature Variables (21 columns)
*All units are in Celsius (°C)*:
5. `Evaporator Water In Temperature Deg C`: Entering chilled water temperature.
6. `Evaporator Water Out Temperature Deg C`: Leaving chilled water temperature (setpoint target).
7. `Condenser Water In Temperature Deg C`: Entering cooling tower water.
8. `Condenser Water Out Temperature Deg C`: Leaving condenser water to cooling tower.
9. `Discharge Temperature Deg C Comp 1`: Compressor 1 discharge refrigerant vapor temp.
10. `Discharge Temperature Deg C Comp 2`: Compressor 2 discharge refrigerant vapor temp.
11. `Suction Temperature Deg C`: Suction line refrigerant gas temp.
12. `Chiler Lift Temperature Deg C Comp 1`: Lift temperature across Comp 1 (Discharge - Suction temp).
13. `Chiler Lift Temperature Deg C Comp 2`: Lift temperature across Comp 2.
14. `Suction Saturation Temperature Deg C`: Saturation temp corresponding to suction pressure.
15. `Suction Superheat deg C`: Suction temperature minus suction saturation temperature.
16. `Evaporator Approach`: Evaporator refrigerant saturation temp vs water out temp.
17. `Discharge Saturation Temperature Deg C Comp 1`: Condenser saturation temp Comp 1.
18. `Discharge Saturation Temperature Deg C Comp 2`: Condenser saturation temp Comp 2.
19. `Discharge Superheat Deg C Comp 1`: Discharge temp minus discharge saturation temp Comp 1.
20. `Discharge Superheat Deg C Comp 2`: Discharge temp minus discharge saturation temp Comp 2.
21. `Condenser Approach Comp 1`: Condenser refrigerant saturation temp vs condenser water out temp.
22. `Condenser Approach Comp 2`: Condenser approach for circuit 2.
23. `Liquid Temperature Deg C Comp 1`: Subcooled liquid refrigerant temperature circuit 1.
24. `Liquid Temperature Deg C Comp 2`: Subcooled liquid refrigerant temperature circuit 2.
25. `Oil Temperature Deg C`: Lubrication oil temperature.

### C. Core Pressure Variables (7 columns)
*All units in kg/cm² ($1\text{ kg/cm}^2 \approx 0.980665\text{ bar} \approx 98.07\text{ kPa}$)*:
26. `Discharge Pressure Kg /cm 2 Comp 1`: High-side refrigerant pressure Comp 1.
27. `Discharge Pressure Kg /cm 2 Comp 2`: High-side refrigerant pressure Comp 2.
28. `Suction Pressure Kg /cm 2`: Low-side suction refrigerant pressure.
29. `Chiller Lift Pressure Kg /cm 2 Comp 1`: Differential pressure across Comp 1 (Discharge - Suction).
30. `Chiller Lift Pressure Kg /cm 2 Comp 2`: Differential pressure across Comp 2.
31. `EXV Differential Pressure Kg /cm 2 Comp 1`: Pressure drop across electronic expansion valve 1.
32. `EXV Differential Pressure Kg /cm 2 Comp 2`: Pressure drop across electronic expansion valve 2.

### D. Compressor Electrical & Load Variables (7 columns)
33. `Running Load % Comp 1`: Percentage load capacity Compressor 1 (0 to 101%).
34. `Running Load % Comp 2`: Percentage load capacity Compressor 2 (0 to 100%).
35. `Motor Current Amps Comp 1`: Current draw Comp 1 (Amperes).
36. `Motor Current Amps Comp 2`: Current draw Comp 2 (Amperes).
37. `Volt R`: Three-phase line voltage Phase R (Volts, ~410-420 V).
38. `Volt Y`: Three-phase line voltage Phase Y (Volts, ~410-420 V).
39. `Volt B`: Three-phase line voltage Phase B (Volts, ~410-420 V).

### E. Actuators, Valves & Speed Drives (3 columns)
40. `VFD Out Comp 1`: Variable Frequency Drive output frequency / percentage.
41. `VFD Out Comp 2`: Variable Frequency Drive output frequency / percentage.
42. `Output To EXV Comp 1`: Electronic Expansion Valve opening percentage (%).

### F. Compressor Accumulated Run Time (2 columns)
43. `Compressure Run Hours Comp 1`: Accumulated operating hours (hours).
44. `Compressure Run Hours Comp 2`: Accumulated operating hours (hours).

### G. Auxiliary Shared Plant Equipment (16 columns)
*Primary Chilled Water Pumps, Condenser Water Pumps, Cooling Tower Fans*:
45. `Primary Pump Suction Pressure Bar`: Suction pressure primary pump (Bar).
46. `Primary Pump Discharge Pressure Bar`: Discharge pressure primary pump (Bar).
47. `Primary Pump Current R Amps`: Current draw Phase R (Amps).
48. `Primary Pump Current Y Amps`: Current draw Phase Y (Amps).
49. `Primary Pump Current B Amps`: Current draw Phase B (Amps).
50. `Condenser Pump Suction Pressure Bar`: Suction pressure condenser pump (Bar).
51. `Condenser Pump Discharge Pressure Bar`: Discharge pressure condenser pump (Bar).
52. `Condenser Pump Current R Amps`: Current draw Phase R (Amps).
53. `Condenser Pump Current Y Amps`: Current draw Phase Y (Amps).
54. `Condenser Pump Current B Amps`: Current draw Phase B (Amps).
55. `Cooling Tower Motor Current Fan 1 R Amps`: Cooling Tower Fan 1 Current R (Amps).
56. `Cooling Tower Motor Current Fan 1 Y Amps`: Cooling Tower Fan 1 Current Y (Amps).
57. `Cooling Tower Motor Current Fan 1 B Amps`: Cooling Tower Fan 1 Current B (Amps).
58. `Cooling Tower Motor Current Fan 2 R Amps`: Cooling Tower Fan 2 Current R (Amps).
59. `Cooling Tower Motor Current Fan 2 Y Amps`: Cooling Tower Fan 2 Current Y (Amps).
60. `Cooling Tower Motor Current Fan 2 B Amps`: Cooling Tower Fan 2 Current B (Amps).

---

## 5. Operating States & Regimes
- **Compressor Configuration:** Each chiller unit is equipped with two semi-hermetic / screw compressors (Comp 1 and Comp 2).
- **Operating Breakdown Across 522 Records:**
  - **Dual Compressor Operation (`Comp 1 > 0` & `Comp 2 > 0`):** 517 observations (99.04%)
  - **Single Compressor Operation (Comp 1 only):** 3 observations (0.57%) — `Sl.No` 185, 335, 483
  - **Single Compressor Operation (Comp 2 only):** 2 observations (0.38%) — `Sl.No` 194, 281
  - **Chiller Offline (`Comp 1 == 0` & `Comp 2 == 0`):** 0 observations (all recorded entries represent running chillers).

---

## 6. Initial Data Quality Assessment & Noteworthy Findings

1. **Missing Values:**
   - There are **0 null / NaN values** in any column after reading with `header=2`. The log sheet is 100% populated.

2. **Temporal Inversion:**
   - Raw observations are recorded in **descending chronological order**. For proper time-series modelling and chronological training/validation/testing splits, observations must be sorted in ascending chronological order (`Timestamp` ascending).

3. **Multi-Unit Plant Composition:**
   - The file aggregates logs from 5 individual chillers into one sheet. Multivariate PCA modelling can either:
     - Model a primary representative chiller (e.g., Chiller-4 with 131 continuous logs or Chiller-3 with 120 logs), or
     - Standardize and train on normal operating periods across homogeneous units, or
     - Fit per-chiller models. (This is a key methodological decision for Phase 2/3).

4. **Industrial Data Entry Typographical Anomalies (Potential Point Anomalies):**
   Several severe anomalies were identified in the raw records, highly indicative of human manual logging typos:
   - **Motor Current Comp 1 (Row 460):** Value = `1350.0 Amps` (at 60% load, expected ~135.0 Amps; 10x decimal shift).
   - **Motor Current Comp 1 (Row 475):** Value = `2196.0 Amps` (at 98% load, expected ~219.6 Amps; 10x decimal shift).
   - **Motor Current Comp 2 (Row 195):** Value = `2106.0 Amps` (at 98% load, expected ~210.6 Amps; 10x decimal shift).
   - **Liquid Temperature Comp 1 (Row 297):** Value = `631.1 °C` (expected ~31.1 °C).
   - **Liquid Temperature Comp 1 (Row 441):** Value = `329.0 °C` (expected ~32.9 °C).
   - **Chiller Lift Temp Comp 2 (Row 391):** Value = `415.0 °C` (expected ~41.5 °C).
   - **Discharge Pressure Comp 1 (Row 213):** Value = `47.9 kg/cm²` (normal ~8.0 kg/cm²).
   - **Discharge Pressure Comp 2 (Row 213):** Value = `88.0 kg/cm²` (normal ~8.0 kg/cm²).
   - **Discharge Pressure Comp 2 (Row 416):** Value = `82.0 kg/cm²` (normal ~8.0 kg/cm²).
   - **Condenser Water Out Temp (Row 388):** Value = `3.0 °C` (Condenser In is 27.1 °C; physically impossible in cooling mode, likely 30.0 °C).

   *Methodological Note:* These existing anomalies will be critical in Phase 2 EDA and Phase 5 detection. In Phase 3, we must decide whether training data should be quarantined/cleaned of these gross typos so PCA learns true normal correlation patterns.

---

## 7. Uncertainties Requiring Student Confirmation
1. **Actuator Units (`VFD Out Comp 1/2` & `Output To EXV Comp 1`):** Headers lack explicit engineering units. Are VFD outputs recorded in Hz (0-50/60 Hz) or % speed (0-100%)? Is EXV output recorded in % opening?
2. **Approach Temperature Units:** `Evaporator Approach`, `Condenser Approach Comp 1`, and `Condenser Approach Comp 2` lack explicit unit labels. Confirm whether these are in Celsius (°C).
3. **Chiller Specification Uniformity:** Confirmation whether Chillers 1 through 5 share identical technical specifications (compressor tonnages, refrigerants) or if significant machine-to-machine capacity variations exist.


