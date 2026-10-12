# Diwali 2026 forecast (pre-registered)

Locked 2026-10-12T16:56:04Z. SHA-256 of the JSON record: `0b5f3a06349b1dd1311300029ef3446796338ac83adc0a3b78a3412ef0120a4f`.

Target: mean hourly (observed - BAU) PM2.5, 18:00 8 Nov to 02:59 9 Nov 2026 IST, missing hours carried forward from the last valid hour at the CPCB Delhi network median, hours with 3 or more reporting sites (config/stations.csv).

| Firecracker rules in 2026 | Forecast (µg/m³) | 80% interval | Based on |
|---|---|---|---|
| Full ban | 366.0 | 232.6 to 499.4 | 2020, 2021 |
| Green crackers in fixed hours | 302.6 | 176.8 to 428.4 | 2018, 2019, 2025 |
| No effective restriction | 191.5 | 58.1 to 324.9 | 2016, 2017 |

80% prediction interval, mean of past Diwalis under the regime +/- t(4 df) x pooled SD (71.0) x sqrt(1 + 1/n)

Scoring: After 8 Nov 2026 the same code is rerun with data through 15 Nov 2026. The forecast for the regime actually in force counts as correct if the observed excess falls inside its interval.

Verify: `sha256sum -c SHA256SUMS` (Linux/macOS) or `Get-FileHash diwali-2026-forecast.json` (Windows PowerShell).
