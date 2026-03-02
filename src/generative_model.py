# src/generative_model.py
import numpy as np


class GenerativeBGAModel:
    """
    Mô hình sinh đơn giản cho BGA:
      - Mỗi population k có vector tần số allele p_{k,j} cho từng SNP j.
      - Genotype g_{ij} ~ Binomial(2, p_{k,j}) (bỏ hệ số tổ hợp).
      - Dùng Bayes để suy ra P(pop | genotype).

    X đầu vào phải là ma trận (n_samples, n_snps) với giá trị 0/1/2 hoặc np.nan (missing).
    """

    def __init__(self, smoothing_alpha: float = 1.0):
        """
        smoothing_alpha: tham số Beta prior (alpha) cho ước lượng p.
                         alpha = 1.0 tương ứng prior Uniform(0,1).
        """
        self.alpha = smoothing_alpha
        self.pop_labels = None     # np.array of strings (pop names)
        self.snp_names = None      # list of snp names (length = n_snps)
        self.allele_freqs = None   # shape (n_pops, n_snps)
        self.priors = None         # shape (n_pops,)

    def fit(self, X: np.ndarray, y, snp_names, priors=None):
        """
        X: np.ndarray shape (n_samples, n_snps), giá trị 0/1/2 hoặc np.nan
        y: array-like, nhãn population (string hoặc int)
        snp_names: list tên SNP tương ứng với cột X
        priors: optional, list/array prior cho từng pop (cùng thứ tự với unique labels).
                Nếu None, dùng prior theo tần suất mẫu (empirical).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        labels, inv = np.unique(y, return_inverse=True)  # labels: unique pops
        n_pops = len(labels)
        n_snps = X.shape[1]

        self.pop_labels = labels
        self.snp_names = list(snp_names)

        # prior
        if priors is None:
            counts = np.bincount(inv)
            self.priors = counts / counts.sum()
        else:
            priors = np.asarray(priors, dtype=float)
            self.priors = priors / priors.sum()

        allele_freqs = np.zeros((n_pops, n_snps), dtype=float)

        # ước lượng p_{k,j} cho từng pop & từng SNP
        for k in range(n_pops):
            mask_k = (inv == k)
            Xk = X[mask_k]  # (n_k, n_snps)

            for j in range(n_snps):
                gj = Xk[:, j]
                gj = gj[~np.isnan(gj)]
                n_valid = len(gj)
                if n_valid == 0:
                    # nếu không có dữ liệu cho SNP này trong pop k, cho p=0.5
                    p = 0.5
                else:
                    alt_count = gj.sum()  # tổng số allele "non-major" (0/1/2)
                    total_chr = 2 * n_valid
                    p = (alt_count + self.alpha) / (total_chr + 2 * self.alpha)

                # tránh p=0 hoặc 1 tuyệt đối để không log(0)
                p = max(min(p, 1 - 1e-4), 1e-4)
                allele_freqs[k, j] = p

        self.allele_freqs = allele_freqs
        return self

    def _log_likelihood_single(self, g_vec: np.ndarray, pop_idx: int) -> float:
        """
        log P(g_vec | pop_idx) = sum_j [ g*log p + (2-g)*log(1-p) ], bỏ hệ số tổ hợp.
        g_vec: shape (n_snps,) với 0/1/2 hoặc np.nan.
        """
        p = self.allele_freqs[pop_idx]  # shape (n_snps,)

        mask = ~np.isnan(g_vec)
        g = g_vec[mask]
        p = p[mask]

        # log p và log (1-p)
        logp = np.log(p)
        log1mp = np.log(1.0 - p)

        # g * log p + (2-g) * log (1-p)
        ll = (g * logp + (2.0 - g) * log1mp).sum()
        return float(ll)

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Trả về log posterior (chuẩn hóa) shape (n_samples, n_pops).
        """
        X = np.asarray(X, dtype=float)
        n_samples, n_snps = X.shape
        n_pops = len(self.pop_labels)

        log_post = np.zeros((n_samples, n_pops), dtype=float)
        log_priors = np.log(self.priors)

        for i in range(n_samples):
            g_vec = X[i]
            for k in range(n_pops):
                ll = self._log_likelihood_single(g_vec, k)
                log_post[i, k] = log_priors[k] + ll

        # log-softmax cho từng sample để ra posterior
        for i in range(n_samples):
            row = log_post[i]
            m = np.max(row)
            row -= m
            probs = np.exp(row)
            probs_sum = probs.sum()
            probs /= probs_sum
            log_post[i] = np.log(probs)

        return log_post

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Posterior probabilities shape (n_samples, n_pops).
        """
        logp = self.predict_log_proba(X)
        return np.exp(logp)

    def predict(self, X: np.ndarray):
        """
        Dự đoán label (không open-set).
        """
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.pop_labels[idx]

    def predict_with_uncertainty(self, X: np.ndarray, threshold: float = 0.7):
        """
        Trả về:
          labels: mảng string, nếu max_prob < threshold -> 'UNKNOWN'
          max_probs: max posterior mỗi sample
          proba: ma trận (n_samples, n_pops)
        """
        proba = self.predict_proba(X)
        max_probs = proba.max(axis=1)
        idx = proba.argmax(axis=1)

        labels = []
        for m, k in zip(max_probs, idx):
            if m < threshold:
                labels.append("UNKNOWN")
            else:
                labels.append(self.pop_labels[k])

        return np.array(labels), max_probs, proba
