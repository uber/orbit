import orbit
from orbit.utils.dataset import load_iclaims
from orbit.models import ETS

print(orbit.__version__)

# load data
df = load_iclaims()
date_col = 'week'
response_col = 'claims'
test_size = 52
train_df = df[:-test_size]
test_df = df[-test_size:]
ets = ETS(
    response_col=response_col,
    date_col=date_col,
    seasonality=52,
    seed=2024,
    estimator="stan-mcmc",
    stan_mcmc_args={
        "show_progress": True,
        "show_console": True
    },
)

predicted_df = ets.predict(df=test_df)
print(predicted_df)