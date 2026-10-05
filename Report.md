

# G-Flix Inc. Data Forensics Investigation Report

**Case Title:** Internal Insider Threat & Data Exfiltration Audit

**Investigator:** Data Forensics Officer

**Target Dataset:** G-Flix User Activity Logs (`anomaly_detection.csv`, 505 records)

---

## 1. Executive Summary

An internal audit of 505 user activity sessions was conducted following suspected breach indicators at G-Flix Inc. To uncover insider threats hidden within standard traffic, an ensemble anomaly detection framework was applied:

* **Statistical Method (Univariate Z-Score, $\vert{}Z\vert{} > 3.0\sigma$):** Flagged extreme standard deviation shifts across core metrics (`data_accessed_MB`, `files_downloaded`, `login_duration_min`).
* **Unsupervised Machine Learning (Isolation Forest, $c=0.015$):** Multi-dimensional partitioning over access volume, download counts, session duration, time of day (`hour`), and remote access flags.

Normal baseline activity is tightly bounded between **0–300 MB** of data accessed and **0–8 file downloads** per session. The audit isolated five primary suspect sessions exhibiting significant divergence from the baseline.

---

## 2. Suspect Dossier & Evidence Breakdown

| Suspect Rank | User ID | Data Accessed (MB) | Files Downloaded | Primary Forensic Evidence |
| --- | --- | --- | --- | --- |
| **#1** | **`user_036`** | **~5000 MB** | **50** | **Critical Bulk Exfiltration:** Exhibits the highest raw data volume accessed across the entire network paired with heavy file downloads, isolated far out in the upper-right quadrant. |
| **#2** | **`user_032`** | **~4000 MB** | **60** | **Mass Directory Extraction:** Substantial volume of internal data accessed combined with the second-highest download count, consistent with unauthorized backup or repository dumping. |
| **#3** | **`user_033`** | **< 100 MB** | **100** | **Automated Scripted Scraping:** Reached the platform-maximum download count (100 files) despite minimal data volume, characteristic of rapid metadata harvesting, credential scraping, or token collection. |
| **#4** | **`user_045`** | **~4500 MB** | **0** | **Direct Database Querying / Streaming:** Massive internal data traversal with zero file downloads, pointing to live database table sniffing, memory dumping, or unmetered media streaming. |
| **#5** | **`user_025`** | **~250 MB** | **4** | **Privilege & Off-Hours Anomaly:** While volume matches standard limits, this session was flagged by the multi-dimensional model due to unauthorized remote access overrides during off-peak/midnight hours. |

---

## 3. Supporting Visual Evidence

* **Behavioral Profiling Map (Left Panel):** Clear spatial separation between baseline activity (dense blue cluster at the bottom left) and accounts `#1`, `#2`, `#3`, and `#4`, which form starkly isolated outliers along the data and download axes.
* **Anomaly Severity Distribution (Right Panel):** The Isolation Forest score distribution places all top suspects past the **0.07 threshold**, directly in the severe right-hand tail of the histogram.

---

## 4. Remediation & Incident Response

1. **Immediate Revocation:** Invalidate active session tokens and disable credentials for accounts `user_036`, `user_032`, `user_033`, and `user_045`.
2. **Access Control Hardening:** Implement rate-limiting rules preventing single-session downloads exceeding **25 files** or aggregate egress over **1000 MB** without secondary multi-factor authentication (MFA).
3. **Endpoint Inspection:** Review authentication source IPs and remote login logs for `user_025` to evaluate potential credential stuffing or session hijacking.