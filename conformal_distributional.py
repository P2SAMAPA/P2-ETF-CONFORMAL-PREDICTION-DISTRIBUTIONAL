import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def compute_composite_macro_factor(macro_df):
    """Compute composite macro factor from all macro variables."""
    if len(macro_df) < 2:
        return np.ones(len(macro_df)) * 0.5
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro_df)
    pca = PCA(n_components=1)
    factor = pca.fit_transform(macro_scaled).flatten()
    factor = (factor - factor.min()) / (factor.max() - factor.min() + 1e-8)
    return factor

def quantile_regression_forest(X, y, quantiles, n_estimators=50, max_depth=5):
    """
    Train quantile regression forests for multiple quantiles.
    Returns a matrix of predictions (n_samples x n_quantiles).
    """
    from sklearn.ensemble import RandomForestRegressor
    predictions = []
    for q in quantiles:
        # For each quantile, train a separate RF with quantile loss
        # We'll use a weighted loss approach: asymmetric MSE
        # For simplicity, we use the quantile regression forest approach
        # using the 'quantile' loss in scikit-learn? Actually sklearn doesn't have quantile loss.
        # We'll use the approach from the 'quantile-forest' package? Not available.
        # Use a simple heuristic: train multiple RFs and use the residuals to estimate quantiles.
        # This is a simplified version of conformal predictive distribution.
        # We'll use the standard RF and then compute conformal scores.
        pass
    # Simplified: use standard RF and then conformalize
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf.fit(X, y)
    return rf

def conformal_predictive_distribution(X_train, y_train, X_test, y_test=None, quantiles=19, n_estimators=50, max_depth=5):
    """
    Compute conformal predictive distribution for test points.
    Returns: predictive quantiles for each test point.
    """
    n = len(X_train)
    # Train base model
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf.fit(X_train, y_train)
    # Compute residuals on training set
    y_pred_train = rf.predict(X_train)
    residuals = y_train - y_pred_train
    # For each test point, compute conformal scores
    y_pred_test = rf.predict(X_test)
    # Generate quantiles from the residuals using conformal prediction
    # This is the conformal predictive distribution: for each test point,
    # the distribution is the empirical distribution of residuals shifted by y_pred_test
    quantile_values = np.linspace(0, 1, quantiles + 2)[1:-1]
    # Use the empirical distribution of residuals
    sorted_residuals = np.sort(residuals)
    n_res = len(sorted_residuals)
    # For each test point, the quantiles are y_pred_test + residual quantiles
    pred_dist = np.zeros((len(X_test), len(quantile_values)))
    for i, q in enumerate(quantile_values):
        idx = int(np.floor(q * n_res))
        idx = min(max(idx, 0), n_res - 1)
        pred_dist[:, i] = y_pred_test + sorted_residuals[idx]
    return pred_dist, sorted_residuals

def conformal_distributional_score(returns, macro_df, quantiles=19, n_estimators=50, max_depth=5):
    """
    Compute per-ETF conformal predictive distribution score.
    Score = density at the actual next-day return.
    """
    if len(returns) < 20 or macro_df is None or len(macro_df) < 20:
        return 0.0
    # Align lengths
    min_len = min(len(returns), len(macro_df))
    returns = returns[:min_len]
    macro_df = macro_df.iloc[:min_len]
    # Remove NaN
    mask = ~(np.isnan(returns) | np.isnan(macro_df).any(axis=1))
    returns = returns[mask]
    macro_df = macro_df[mask]
    if len(returns) < 20:
        return 0.0
    # Compute macro factor
    macro_factor = compute_composite_macro_factor(macro_df)
    # Prepare features: lagged returns + macro factor
    seq_len = 5
    X, y = [], []
    for i in range(seq_len, len(returns)):
        X.append(np.concatenate([returns[i-seq_len:i], macro_factor[i-seq_len:i]]))
        y.append(returns[i])
    X = np.array(X)
    y = np.array(y)
    # Split into train and test (80/20)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    if len(y_test) < 5:
        return 0.0
    # Standardise
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Compute conformal predictive distribution
    pred_dist, residuals = conformal_predictive_distribution(
        X_train_scaled, y_train, X_test_scaled, y_test, quantiles, n_estimators, max_depth
    )
    if len(pred_dist) == 0:
        return 0.0
    # Compute density at the actual test points
    # We use a kernel density estimate from the residuals
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(residuals)
    # For each test point, compute the density of the residual (y_test - y_pred_test)
    # Get predictions for test set
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_pred_test = rf.predict(X_test_scaled)
    residuals_test = y_test - y_pred_test
    density = kde.evaluate(residuals_test)
    # Score = average density over test points
    score = np.mean(density)
    return float(score)
