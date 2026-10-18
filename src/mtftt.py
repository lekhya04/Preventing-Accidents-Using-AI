"""
Multi-Task FT-Transformer with Uncertainty-Weighted Loss  (pure NumPy implementation)

Adopted from:  Y. Wang, Z. Liang, Y. He, "Multi-Task FT-Transformer: A High-Performance and
Interpretable Framework for Traffic Accident Severity Prediction", IEEE Access, 2026.

Pieces implemented exactly as described in the paper:
  * Feature Tokenizer   E_i = b_i + x_i * W_i          (numeric features)
                        E_i = b_i + e_i^T W_i          (categorical features, embedding lookup)
  * [CLS] token prepended, Pre-Norm Transformer layer(s) (multi-head self-attention + FFN)
  * Prediction head     y_k = Linear_k(ReLU(LayerNorm(T_L[CLS])))   - one head per task
  * Homoscedastic-uncertainty task weighting
                        L = sum_k  exp(-s_k) * CE_k + 0.5 * s_k     with s_k = log(sigma_k^2)

Written in NumPy (manual back-propagation + Adam) so the project runs on any laptop without
PyTorch. Gradients are verified numerically in `gradient_check()`.
"""
import numpy as np


def _ln_fwd(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    xh = (x - mu) / np.sqrt(var + eps)
    return xh * g + b, (xh, var, eps)


def _ln_bwd(dy, cache, g):
    xh, var, eps = cache
    d = xh.shape[-1]
    dg = (dy * xh).reshape(-1, d).sum(0)
    db = dy.reshape(-1, d).sum(0)
    dxh = dy * g
    inv = 1.0 / np.sqrt(var + eps)
    dx = inv / d * (d * dxh - dxh.sum(-1, keepdims=True) - xh * (dxh * xh).sum(-1, keepdims=True))
    return dx, dg, db


def _softmax(z, axis=-1):
    z = z - z.max(axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis, keepdims=True)


class MultiTaskFTTransformer:
    def __init__(self, cat_cardinalities, n_num, task_classes, d=24, n_heads=2, n_layers=1,
                 ffn_mult=2, seed=0):
        rng = np.random.default_rng(seed)
        self.cards = list(cat_cardinalities)
        self.n_num = n_num
        self.tasks = list(task_classes)
        self.d, self.h, self.L = d, n_heads, n_layers
        assert d % n_heads == 0
        s = 1.0 / np.sqrt(d)
        P = {}
        # feature tokenizer
        P["W_num"] = rng.normal(0, s, (n_num, d)) if n_num else np.zeros((0, d))
        P["b_num"] = np.zeros((n_num, d))
        self.cat_offsets = np.concatenate([[0], np.cumsum(self.cards)[:-1]]).astype(int) if self.cards else np.zeros(0, int)
        P["E_cat"] = rng.normal(0, s, (int(sum(self.cards)), d)) if self.cards else np.zeros((0, d))
        P["b_cat"] = np.zeros((len(self.cards), d))
        P["cls"] = rng.normal(0, s, (d,))
        for l in range(n_layers):
            for nm in ("q", "k", "v", "o"):
                P[f"W{nm}{l}"] = rng.normal(0, s, (d, d))
            P[f"ln1g{l}"], P[f"ln1b{l}"] = np.ones(d), np.zeros(d)
            P[f"ln2g{l}"], P[f"ln2b{l}"] = np.ones(d), np.zeros(d)
            P[f"W1{l}"] = rng.normal(0, s, (d, d * ffn_mult)); P[f"b1{l}"] = np.zeros(d * ffn_mult)
            P[f"W2{l}"] = rng.normal(0, 1.0 / np.sqrt(d * ffn_mult), (d * ffn_mult, d)); P[f"b2{l}"] = np.zeros(d)
        P["lnhg"], P["lnhb"] = np.ones(d), np.zeros(d)
        for k, c in enumerate(self.tasks):
            P[f"Wh{k}"] = rng.normal(0, s, (d, c)); P[f"bh{k}"] = np.zeros(c)
        P["log_var"] = np.zeros(len(self.tasks))     # s_k = log sigma_k^2  (learnable)
        self.P = P
        self.m = {k: np.zeros_like(v) for k, v in P.items()}
        self.v = {k: np.zeros_like(v) for k, v in P.items()}
        self.t = 0

    # ------------------------------------------------------------------ forward
    def forward(self, Xc, Xn, keep_attn=False):
        P, d, H = self.P, self.d, self.h
        B = Xc.shape[0] if Xc.size else Xn.shape[0]
        toks = []
        if self.n_num:
            toks.append(Xn[:, :, None] * P["W_num"][None] + P["b_num"][None])
        if self.cards:
            idx = Xc + self.cat_offsets[None]
            toks.append(P["E_cat"][idx] + P["b_cat"][None])
        E = np.concatenate(toks, 1)
        T = np.concatenate([np.broadcast_to(P["cls"], (B, 1, d)), E], 1)
        caches = {"Xc": Xc, "Xn": Xn, "layers": []}
        attn_maps = []
        for l in range(self.L):
            c = {}
            c["T_in"] = T
            Hn, c["ln1"] = _ln_fwd(T, P[f"ln1g{l}"], P[f"ln1b{l}"])
            c["Hn"] = Hn
            Q, K, V = Hn @ P[f"Wq{l}"], Hn @ P[f"Wk{l}"], Hn @ P[f"Wv{l}"]
            n = T.shape[1]; dh = d // H
            sp = lambda M: M.reshape(B, n, H, dh).transpose(0, 2, 1, 3)
            Qh, Kh, Vh = sp(Q), sp(K), sp(V)
            S = Qh @ Kh.transpose(0, 1, 3, 2) / np.sqrt(dh)
            A = _softmax(S)
            Oh = A @ Vh
            O = Oh.transpose(0, 2, 1, 3).reshape(B, n, d)
            Z = O @ P[f"Wo{l}"]
            T1 = T + Z
            H2, c["ln2"] = _ln_fwd(T1, P[f"ln2g{l}"], P[f"ln2b{l}"])
            pre = H2 @ P[f"W1{l}"] + P[f"b1{l}"]
            R = np.maximum(pre, 0)
            U = R @ P[f"W2{l}"] + P[f"b2{l}"]
            T = T1 + U
            c.update(Qh=Qh, Kh=Kh, Vh=Vh, A=A, O=O, H2=H2, pre=pre, R=R, dh=dh)
            caches["layers"].append(c)
            if keep_attn:
                attn_maps.append(A)
        cls = T[:, 0]
        hn, caches["lnh"] = _ln_fwd(cls, P["lnhg"], P["lnhb"])
        g = np.maximum(hn, 0)
        logits = [g @ P[f"Wh{k}"] + P[f"bh{k}"] for k in range(len(self.tasks))]
        caches.update(T_out=T, hn=hn, g=g)
        self._cache = caches
        if keep_attn:
            self._attn = attn_maps
        return logits

    def loss(self, logits, Y):
        """Uncertainty-weighted multi-task cross-entropy. Y: (B, n_tasks) int labels."""
        s = self.P["log_var"]
        total, parts, probs = 0.0, [], []
        for k, lg in enumerate(logits):
            p = _softmax(lg)
            ce = -np.log(p[np.arange(len(p)), Y[:, k]] + 1e-12).mean()
            parts.append(ce); probs.append(p)
            total += np.exp(-s[k]) * ce + 0.5 * s[k]
        self._probs = probs
        return total, parts

    # ------------------------------------------------------------------ backward
    def backward(self, Y, class_weights=None):
        P, c0 = self.P, self._cache
        G = {k: np.zeros_like(v) for k, v in P.items()}
        B = Y.shape[0]
        s = P["log_var"]
        dg = np.zeros_like(c0["g"])
        for k, p in enumerate(self._probs):
            ce = -np.log(p[np.arange(B), Y[:, k]] + 1e-12).mean()
            G["log_var"][k] = -np.exp(-s[k]) * ce + 0.5
            dl = p.copy(); dl[np.arange(B), Y[:, k]] -= 1; dl *= np.exp(-s[k]) / B
            G[f"Wh{k}"] = c0["g"].T @ dl; G[f"bh{k}"] = dl.sum(0)
            dg += dl @ P[f"Wh{k}"].T
        dhn = dg * (c0["hn"] > 0)
        dcls, G["lnhg"], G["lnhb"] = _ln_bwd(dhn, c0["lnh"], P["lnhg"])
        dT = np.zeros_like(c0["T_out"]); dT[:, 0] = dcls
        d, H = self.d, self.h
        for l in reversed(range(self.L)):
            c = c0["layers"][l]
            n = dT.shape[1]; dh = c["dh"]
            # FFN
            dU = dT
            G[f"W2{l}"] = c["R"].reshape(-1, c["R"].shape[-1]).T @ dU.reshape(-1, d); G[f"b2{l}"] = dU.reshape(-1, d).sum(0)
            dR = dU @ P[f"W2{l}"].T
            dpre = dR * (c["pre"] > 0)
            G[f"W1{l}"] = c["H2"].reshape(-1, d).T @ dpre.reshape(-1, dpre.shape[-1]); G[f"b1{l}"] = dpre.reshape(-1, dpre.shape[-1]).sum(0)
            dH2 = dpre @ P[f"W1{l}"].T
            dT1_ln, G[f"ln2g{l}"], G[f"ln2b{l}"] = _ln_bwd(dH2, c["ln2"], P[f"ln2g{l}"])
            dT1 = dT + dT1_ln
            # attention
            dZ = dT1
            G[f"Wo{l}"] = c["O"].reshape(-1, d).T @ dZ.reshape(-1, d)
            dO = dZ @ P[f"Wo{l}"].T
            dOh = dO.reshape(B, n, H, dh).transpose(0, 2, 1, 3)
            dA = dOh @ c["Vh"].transpose(0, 1, 3, 2)
            dVh = c["A"].transpose(0, 1, 3, 2) @ dOh
            dS = c["A"] * (dA - (dA * c["A"]).sum(-1, keepdims=True)) / np.sqrt(dh)
            dQh = dS @ c["Kh"]
            dKh = dS.transpose(0, 1, 3, 2) @ c["Qh"]
            mg = lambda M: M.transpose(0, 2, 1, 3).reshape(B, n, d)
            dQ, dK, dV = mg(dQh), mg(dKh), mg(dVh)
            Hn = c["Hn"].reshape(-1, d)
            G[f"Wq{l}"] = Hn.T @ dQ.reshape(-1, d); G[f"Wk{l}"] = Hn.T @ dK.reshape(-1, d); G[f"Wv{l}"] = Hn.T @ dV.reshape(-1, d)
            dHn = dQ @ P[f"Wq{l}"].T + dK @ P[f"Wk{l}"].T + dV @ P[f"Wv{l}"].T
            dT_ln, G[f"ln1g{l}"], G[f"ln1b{l}"] = _ln_bwd(dHn, c["ln1"], P[f"ln1g{l}"])
            dT = dT1 + dT_ln
        G["cls"] = dT[:, 0].sum(0)
        dE = dT[:, 1:]
        j = 0
        if self.n_num:
            dEn = dE[:, :self.n_num]
            G["W_num"] = (dEn * c0["Xn"][:, :, None]).sum(0); G["b_num"] = dEn.sum(0); j = self.n_num
        if self.cards:
            dEc = dE[:, j:]
            G["b_cat"] = dEc.sum(0)
            idx = (c0["Xc"] + self.cat_offsets[None]).ravel()
            np.add.at(G["E_cat"], idx, dEc.reshape(-1, d))
        return G

    def step(self, G, lr=1e-3, b1=0.9, b2=0.999, wd=1e-5):
        self.t += 1
        for k in self.P:
            g = G[k] + (wd * self.P[k] if k not in ("log_var",) and not k.startswith("ln") else 0)
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1 ** self.t); vh = self.v[k] / (1 - b2 ** self.t)
            self.P[k] -= lr * mh / (np.sqrt(vh) + 1e-8)

    # ------------------------------------------------------------------ API
    def fit(self, Xc, Xn, Y, Xc_val=None, Xn_val=None, Y_val=None, epochs=8, batch=512, lr=2e-3,
            patience=3, seed=0, log=print):
        rng = np.random.default_rng(seed)
        n = len(Y); best, best_P, wait, hist = np.inf, None, 0, []
        for ep in range(epochs):
            perm = rng.permutation(n); tot = 0.0
            for i in range(0, n, batch):
                b = perm[i:i + batch]
                lg = self.forward(Xc[b], Xn[b])
                L, _ = self.loss(lg, Y[b]); tot += L * len(b)
                self.step(self.backward(Y[b]), lr=lr)
            rec = {"epoch": ep + 1, "train_loss": tot / n, "sigma": np.exp(0.5 * self.P["log_var"]).round(3).tolist()}
            if Y_val is not None:
                vl = self.loss(self.forward(Xc_val, Xn_val), Y_val)[0]
                rec["val_loss"] = float(vl)
                if vl < best - 1e-4:
                    best, wait = vl, 0; best_P = {k: v.copy() for k, v in self.P.items()}
                else:
                    wait += 1
            hist.append(rec); log(rec)
            if Y_val is not None and wait >= patience:
                break
        if best_P is not None:
            self.P = best_P
        return hist

    def predict_proba(self, Xc, Xn, batch=4096):
        outs = [[] for _ in self.tasks]
        n = Xc.shape[0] if Xc.size else Xn.shape[0]
        for i in range(0, n, batch):
            lg = self.forward(Xc[i:i + batch], Xn[i:i + batch])
            for k, l in enumerate(lg):
                outs[k].append(_softmax(l))
        return [np.concatenate(o) for o in outs]

    def attention_importance(self, Xc, Xn):
        """Mean CLS->feature attention (last layer, averaged over heads) = per-feature importance."""
        self.forward(Xc, Xn, keep_attn=True)
        A = self._attn[-1]                     # (B, H, n, n)
        return A[:, :, 0, 1:].mean((0, 1))


def gradient_check(seed=1):
    rng = np.random.default_rng(seed)
    m = MultiTaskFTTransformer([4, 3], 2, [3, 2], d=8, n_heads=2, n_layers=1, seed=seed)
    for k in m.P:
        m.P[k] = m.P[k] + rng.normal(0, 0.1, m.P[k].shape)
    Xc = rng.integers(0, 3, (5, 2)); Xn = rng.normal(size=(5, 2)); Y = np.stack([rng.integers(0, 3, 5), rng.integers(0, 2, 5)], 1)
    m.loss(m.forward(Xc, Xn), Y); G = m.backward(Y)
    worst = 0.0
    for k in m.P:
        flat = m.P[k].ravel()
        for i in rng.choice(flat.size, min(4, flat.size), replace=False):
            old = flat[i]; eps = 1e-5
            flat[i] = old + eps; lp = m.loss(m.forward(Xc, Xn), Y)[0]
            flat[i] = old - eps; lm = m.loss(m.forward(Xc, Xn), Y)[0]
            flat[i] = old
            num = (lp - lm) / (2 * eps); ana = G[k].ravel()[i]
            rel = abs(num - ana) / max(1e-7, abs(num) + abs(ana))
            worst = max(worst, rel)
    return worst


if __name__ == "__main__":
    print("max relative gradient error:", gradient_check())
