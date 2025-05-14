    def _predict_with_psf(self, data, n_ahead):
        """
        Predict future values using PSF algorithm
        """
        # Remove NaNs and convert to numpy array
        data = np.array(data)
        data = data[~np.isnan(data)]
        
        # If data is too short, use ARIMA directly
        if len(data) < self.cycle * 2:
            try:
                model = ARIMA(data, order=(1, 1, 1))
                fit = model.fit()
                return fit.forecast(n_ahead)
            except Exception as e:
                print(f"ARIMA failed: {e}")
                return np.zeros(n_ahead)
        
        xn = len(data)
        y = xn % self.cycle
        x1 = data[y:xn]

        try:
            # Convert to pandas Series for PSF
            x1_series = pd.Series(x1)
            # Replace any infinite values with NaN
            x1_series = x1_series.replace([np.inf, -np.inf], np.nan)
            # Fill NaN values with forward fill and backward fill
            x1_series = x1_series.fillna(method='ffill').fillna(method='bfill')
            
            # Calculate number of clusters based on data length
            n_clusters = max(2, min(5, len(x1_series) // self.cycle))
            
            psf_model = Psf(cycle_length=self.cycle, 
                           apply_diff=True, 
                           diff_periods=self.cycle,
                           n_clusters=n_clusters)
            psf_model.fit(x1_series)
            predictions = psf_model.predict(n_ahead=self.cycle)
            return predictions[:n_ahead]
        except Exception as e:
            print(f"PSF failed: {e}")
            try:
                # Fallback to ARIMA
                model = ARIMA(data, order=(1, 1, 1))
                fit = model.fit()
                return fit.forecast(n_ahead)
            except Exception as e:
                print(f"ARIMA failed: {e}")
                return np.zeros(n_ahead) 