# import matplotlib.pyplot as plt
# import orbit
# # from orbit.template.dlt import DLTModel
# from orbit.models import DLT
# from orbit.utils.dataset import load_iclaims

# def run_dlt_pystan(df, response_col, date_col):
#     # load log-transformed data
#     dlt = DLT(
#         seasonality=52,
#         seasonality_sm_input=0.1,
#         level_sm_input=0.1,
#         response_col=response_col,
#         date_col=date_col,
#         estimator='stan-map',
#     )
#     # expect an error message if pystan is not installed
#     dlt.fit(df)
# # response_col = 'claims'
# # date_col = 'week'


# if __name__ == "main":
