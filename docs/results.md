| Method              | Variant type | I-A Recall   | I-A Precision | I-A F1-score | I-B Recall   | I-B Precision | I-B F1-score |
|---------------------|--------------|----------|-----------|----------|----------|-----------|----------|
| BaselineBN          | Indel        | 94.06%   | 87.21%    | 90.51%   | 66.22%   | 48.06%    | 55.70%   |
|                     | SNP          | 99.14%   | 99.56%    | 99.35%   | 90.76%   | 80.12%    | 85.11%   |
| RUN-DVC             | Indel        | 94.65%   | 92.28%    | 93.45%   | 61.22%   | 69.42%    | 65.06%   |
|                     | SNP          | 99.17%   | 99.53%    | 99.35%   | 88.54%   | 94.69%    | 91.51%   |
| Full-label          | Indel        | 96.70%   | 97.58%    | 97.14%   | 64.62%   | 86.07%    | 73.82%   |
|                     | SNP          | 99.14%   | 99.64%    | 99.39%   | 89.33%   | 97.01%    | 93.02%   |

*Table Caption*: Variant calling result (PASS calls) in HG002 sample of I-A and I-B under UDA setting.

---

| Method         | Variant type | I-C Recall   | I-C Precision | I-C F1-score | I-D Recall   | I-D Precision | I-D F1-score |
|----------------|--------------|----------|-----------|----------|----------|-----------|----------|
| BaselineBN     | Indel        | 86.29%   | 92.85%    | 89.45%   | 91.42%   | 82.87%    | 86.93%   |
|                | SNP          | 95.50%   | 99.44%    | 97.43%   | 98.81%   | 99.65%    | 99.23%   |
| RUN-DVC        | Indel        | 87.35%   | 95.27%    | 91.14%   | 94.71%   | 93.85%    | 94.28%   |
|                | SNP          | 95.49%   | 99.52%    | 97.47%   | 99.04%   | 99.70%    | 99.37%   |
| Full-label     | Indel        | 89.88%   | 97.61%    | 93.59%   | 96.37%   | 97.19%    | 96.78%   |
|                | SNP          | 96.20%   | 99.52%    | 97.83%   | 99.22%   | 99.69%    | 99.45%   |

*Table Caption*: Variant calling result (PASS calls) in HG002 sample of I-C and I-D under UDA setting.

---

| Method       | Variant type | P-A Recall | P-A Precision | P-A F1-score | O-A Recall | O-A Precision | O-A F1-score |
|--------------|--------------|------------|---------------|--------------|------------|---------------|--------------|
| BaselineBN   | Indel        | 97.14%     | 96.22%        | 96.68%       | 74.01%     | 80.42%        | 77.08%       |
|              | SNP          | 99.39%     | 99.93%        | 99.66%       | 99.61%     | 99.59%        | 99.60%       |
| RUN-DVC      | Indel        | 97.44%     | 97.03%        | 97.23%       | 73.87%     | 81.21%        | 77.37%       |
|              | SNP          | 99.42%     | 99.93%        | 99.68%       | 99.60%     | 99.58%        | 99.59%       |
| Full-label   | Indel        | 98.48%     | 98.69%        | 98.58%       | 76.40%     | 83.72%        | 79.89%       |
|              | SNP          | 99.56%     | 99.93%        | 99.75%       | 99.63%     | 99.58%        | 99.60%       |

*Table Caption*: Variant calling result (PASS calls) in HG002 sample of P-A and O-A under UDA setting.

<img src="fig2-a.jpg" width="80%" style="margin-left: auto; margin-right: auto; display: block;"/>


<!-- | Method              | Variant type | I-A Recall   | I-A Precision | I-A F1-score | I-B Recall   | I-B Precision | I-B F1-score |
|---------------------|--------------|----------|-----------|----------|----------|-----------|----------|
| BaselineBN          | Indel        | 94.06%   | 87.21%    | 90.51%   | 66.22%   | 48.06%    | 55.70%   |
|                     | SNP          | 99.14%   | 99.56%    | 99.35%   | 90.76%   | 80.12%    | 85.11%   |
| Clair3              | Indel        | 94.53%   | 87.18%    | 90.71%   | 67.93%   | 41.68%    | 51.66%   |
|                     | SNP          | 99.18%   | 99.52%    | 99.35%   | 91.75%   | 57.94%    | 71.03%   |
| DeepVariant         | Indel        | 96.83%   | 96.56%    | 96.69%   | 66.91%   | 55.92%    | 60.92%   |
|                     | SNP          | 99.20%   | 99.84%    | 99.52%   | 90.85%   | 94.21%    | 92.50%   |
| RUN-DVC             | Indel        | 94.65%   | 92.28%    | 93.45%   | 61.22%   | 69.42%    | 65.06%   |
|                     | SNP          | 99.17%   | 99.53%    | 99.35%   | 88.54%   | 94.69%    | 91.51%   |
| Full-label          | Indel        | 96.70%   | 97.58%    | 97.14%   | 64.62%   | 86.07%    | 73.82%   |
|                     | SNP          | 99.14%   | 99.64%    | 99.39%   | 89.33%   | 97.01%    | 93.02%   |

*Table Caption*: Variant calling result (PASS calls) in HG002 sample of I-A and I-B under UDA setting.

---

| Method         | Variant type | I-C Recall   | I-C Precision | I-C F1-score | I-D Recall   | I-D Precision | I-D F1-score |
|----------------|--------------|----------|-----------|----------|----------|-----------|----------|
| BaselineBN     | Indel        | 86.29%   | 92.85%    | 89.45%   | 91.42%   | 82.87%    | 86.93%   |
|                | SNP          | 95.50%   | 99.44%    | 97.43%   | 98.81%   | 99.65%    | 99.23%   |
| Clair3         | Indel        | 88.63%   | 92.93%    | 90.73%   | 94.05%   | 85.94%    | 89.81%   |
|                | SNP          | 95.93%   | 99.38%    | 97.62%   | 99.32%   | 99.40%    | 99.36%   |
| DeepVariant    | Indel        | 86.96%   | 95.91%    | 91.22%   | 96.81%   | 97.58%    | 97.19%   |
|                | SNP          | 89.20%   | 99.67%    | 94.15%   | 99.27%   | 99.81%    | 99.54%   |
| RUN-DVC        | Indel        | 87.35%   | 95.27%    | 91.14%   | 94.71%   | 93.85%    | 94.28%   |
|                | SNP          | 95.49%   | 99.52%    | 97.47%   | 99.04%   | 99.70%    | 99.37%   |
| Full-label     | Indel        | 89.88%   | 97.61%    | 93.59%   | 96.37%   | 97.19%    | 96.78%   |
|                | SNP          | 96.20%   | 99.52%    | 97.83%   | 99.22%   | 99.69%    | 99.45%   |

*Table Caption*: Variant calling result (PASS calls) in HG002 sample of I-C and I-D under UDA setting.

---

| Method       | Variant type | P-A Recall | P-A Precision | P-A F1-score | O-A Recall | O-A Precision | O-A F1-score |
|--------------|--------------|------------|---------------|--------------|------------|---------------|--------------|
| BaselineBN   | Indel        | 97.14%     | 96.22%        | 96.68%       | 74.01%     | 80.42%        | 77.08%       |
|              | SNP          | 99.39%     | 99.93%        | 99.66%       | 99.61%     | 99.59%        | 99.60%       |
| Clair3       | Indel        | 97.29%     | 95.34%        | 96.31%       | 71.32%     | 85.46%        | 77.75%       |
|              | SNP          | 99.88%     | 99.92%        | 99.90%       | 99.62%     | 99.66%        | 99.64%       |
| DeepVariant  | Indel        | 97.31%     | 97.21%        | 97.26%       | 73.40%     | 83.10%        | 77.95%       |
|              | SNP          | 99.90%     | 99.95%        | 99.93%       | 99.66%     | 99.76%        | 99.71%       |
| RUN-DVC      | Indel        | 97.44%     | 97.03%        | 97.23%       | 73.87%     | 81.21%        | 77.37%       |
|              | SNP          | 99.42%     | 99.93%        | 99.68%       | 99.60%     | 99.58%        | 99.59%       |
| Full-label   | Indel        | 98.48%     | 98.69%        | 98.58%       | 76.40%     | 83.72%        | 79.89%       |
|              | SNP          | 99.56%     | 99.93%        | 99.75%       | 99.63%     | 99.58%        | 99.60%       |

*Table Caption*: Variant calling result (PASS calls) in HG002 sample of P-A and O-A under UDA setting. -->
