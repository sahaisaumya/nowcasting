import datetime
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.metrics import r2_score
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')

def plot_day_trend(date, main_df):
	'''
	Routine to plot figure 2 in the paper.
	'''
	plt.figure(figsize=(5,3), dpi = 150, edgecolor='b')
	plt.plot(main_df[date], marker='.', color="indianred", markersize=6, lw=0.5)
	plt.xlabel(r"$T$", fontsize=12,)
	plt.ylabel(r"$I_{TD}$", fontsize = 12,)
	plt.xticks(rotation=0, fontsize=8)
	plt.yticks(rotation=0, fontsize=8)
	plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=6))
	plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %d, %y"))
	plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6))
	plt.ylim(-100, None)
	plt.grid(b=False)
	plt.title(r'D = '+date, size=9)
	sns.despine()

def plot_total_case_hist(agg_pred_data):
	'''
	Routine to plot figure 1 in the paper.
	'''
	plt.figure(figsize = (10,4))
	start = mdates.date2num(agg_pred_data.index[-15])
	end = mdates.date2num(agg_pred_data.index[-1])
	width = end - start
	rect = Rectangle((start, 0), width, max(agg_pred_data['daily_confirm']), color='gray', alpha=0.4)
	plt.gca().add_patch(rect)
	plt.bar(agg_pred_data.index, agg_pred_data['daily_confirm'], width=1.2, color='indianred')
	plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %d, %y"))
	plt.xlim([datetime.date(2020, 3, 1), None])
	plt.yticks( size=12, visible=True)
	plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6))
	plt.grid(b=False)
	plt.ylabel("Number of infections", fontsize=12)
	sns.despine(bottom=False)

def plot_actual_predicted(y_train, pred):
	'''
	Routine to plot figure 3 in the paper.
	'''
	plt.figure(figsize=(5,3), dpi = 175)
	plt.scatter(y_train, pred, alpha=0.8, color = "teal", s=4)
	plt.xlabel('Actual '+r'$F_{td}$')
	plt.ylabel("Predicted "+ r'$F_{td}$')
	plt.xlim(None, 1.02)
	plt.ylim(None, 1.02)
	plt.xticks([0,.25,0.5,0.75,1], fontsize=8)
	plt.yticks([0.25,0.5,0.75,1], fontsize=8)
	plt.grid(b=False)
	sns.despine()

def days_since_vs_missingness(pred, y_train, subset_df_p, days_to_plot = 14):
	'''
	Routine to plot figure 4 in the paper.
	'''
	fig = plt.figure(figsize=(6,2), dpi = 150)
	err = np.sqrt(sum((pred - y_train)**2) / (len(y_train) - 2))
	for day in range(0,days_to_plot):
		day_since = day
		sub = subset_df_p[subset_df_p.day_since_first == day_since].index
		if day == 0:
			plt.errorbar(np.linspace(day, day+0.8,len(subset_df_p.iloc[sub].level_1)), pred[sub], yerr=err, color='teal', fmt='.', alpha = 0.3, elinewidth=0.8, markersize=8,label='prediction')
			plt.plot(np.linspace(day, day+0.8,len(subset_df_p.iloc[sub].level_1)), y_train[sub], '.', color = 'indianred', alpha=0.8, markersize=6, label='actual')
		else:
			plt.errorbar(np.linspace(day, day+0.8,len(subset_df_p.iloc[sub].level_1)), pred[sub], yerr=err, color='teal', fmt='.', alpha = 0.3,  elinewidth=0.8, markersize=8)
			plt.plot(np.linspace(day, day+0.8,len(subset_df_p.iloc[sub].level_1)), y_train[sub], '.', color = 'indianred', alpha=0.8, markersize=6)	
	plt.legend()
	plt.xlabel(r"$ \Delta_{td}$", fontsize=12)
	plt.ylabel(r'$F_{td}$', fontsize = 12)
	plt.xticks( np.linspace(0.4,days_to_plot+0.4-1, len(range(0,days_to_plot))),range(0,days_to_plot), fontsize=8)
	plt.yticks([0,.5, 1], fontsize=8)
	plt.ylim((-0.1,1.2))
	plt.xlim(-0.1,days_to_plot)
	fig.tight_layout()
	plt.grid(b=False)
	sns.despine()

def backfill_data(days, agg_pred_data, clf):
	'''
	This functions uses the fitted model (clf) to nowcast the data file (agg_pred_data) from last day of data for 'days' number of days 
	'''
	nowcasted_data = agg_pred_data.copy()
	for date in nowcasted_data.index[-days:]:

		day_collection = datetime.datetime(date.year, date.month, date.day).weekday()
		day_since_coll_ = ( nowcasted_data.index[-1] - date).days
		assert day_since_coll_ >= 0

		predictionarray =  (list(pd.get_dummies(6).reindex(columns=range(0, 7), fill_value=0).values[0]))
		predictionarray.extend([day_since_coll_, nowcasted_data["daily_confirm"][date],day_since_coll_**2, day_since_coll_**3 ])

		sq_inf = clf.predict([predictionarray])
		sq_inf = 1 - (sq_inf) 

		nowcasted_data["daily_confirm"][date] = int(nowcasted_data["daily_confirm"][date]/sq_inf )
		
	nowcasted_data["cum_confirm"] = nowcasted_data["daily_confirm"].cumsum()
	nowcasted_data.index.name = 'time'

	return nowcasted_data

def plot_explained_bar(score, txt):
	'''
	This functions makes a bar for showing explained variance of the model.
	'''
	plt.figure(figsize=(5,0.2), dpi=150)
	plt.yticks([])
	plt.text(x = score-0.01, y= 0.75, s = f'{score:.2f}', fontsize=5, horizontalalignment='left')
	plt.xticks([0,1], fontsize=5, rotation=0)
	
	plt.barh(0, score, lw=1, fc='teal', fill=True, align = 'center')
	plt.barh(0, 1, lw=1, ec='black', fill=False, align = 'center')
	plt.title(f"Explained variance for {txt} data", fontsize=8, y=1.7)
	sns.despine(bottom=True, left=True)

def plot_sample_corrected_day(date, main_df, clf, thresh):
	'''
	Routine to plot figure 5 in the paper.
	'''
	subset_df = main_df.loc[:,date:date]
	subset_df_p = generate_fit_data(subset_df)
	subset_df_p = subset_df_p.rename(columns={0: 'infections'})
	final = subset_df_p.drop(["collection_date","level_1"], axis=1)
	X_train = pd.concat([pd.get_dummies(final["day_collection"]).reindex(columns=range(0, 7), fill_value=0),final['day_since_first'], final['infections'], final['day_since_first_quad'], final['day_since_first_cub']], axis=1)

	sq_inf = clf.predict(X_train)
	sq_inf = 1 - (sq_inf) 
	sq_inf_final = sq_inf[sq_inf<(1-thresh)].reshape(-1,1)

	er = np.sqrt(np.mean((max(subset_df.dropna().values)  - subset_df.dropna().values )**2))
	plt.figure(dpi=200, figsize=(5,3))
	plt.axhline(max(subset_df.dropna().values), color='k', linestyle='--', label=r'$I^s_D$ (stable)')
	plt.plot((subset_df.dropna().values), label = "actual", marker='.', color='indianred', lw=1.5, ms=8, alpha=0.9)

	plt.errorbar(x=range(len(subset_df.dropna()[sq_inf<(1-thresh)].index)),y=(subset_df.dropna()[sq_inf<(1-thresh)].values/sq_inf_final),  label="prediction", marker = '.' ,color='teal', alpha=0.9)
	plt.fill_between(x=range(len(subset_df.dropna()[sq_inf<(1-thresh)].index)), y1=np.ndarray.flatten((subset_df.dropna()[sq_inf<(1-thresh)].values/sq_inf_final)+er), y2=np.ndarray.flatten((subset_df.dropna()[sq_inf<(1-thresh)].values/sq_inf_final)-er), color='teal', alpha=0.2)
	plt.title(r'$D = $'+date, size=9)
	plt.ylabel(r"$I_{tD}$")
	plt.xlabel(r"$\Delta_{tD}$")
	plt.yticks( fontsize=8)
	plt.xticks(fontsize=8)
	plt.grid(b=False)
	plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
	plt.legend(loc="lower right", prop={'size': 8})
	sns.despine()

def load_nowcast_data(path):
	'''
	Loads the data file that is to be nowcast.
	'''
	pred_data = pd.read_csv(path)[:-1]
	pred_data['Onset Date'] = pd.to_datetime(pred_data['Onset Date'], format="%m/%d/%Y")
	pred_data = pred_data.sort_values(by = ['Onset Date'])
	pred_data = pred_data.astype({'Case Count': 'int32'})
	pred_data = pred_data.astype({'Death Due to Illness Count': 'int32'})
	agg_pred_data = pred_data.groupby('Onset Date').agg({'Case Count': 'sum'})
	agg_pred_data["deaths"] = pred_data.groupby('Onset Date').agg({'Death Due to Illness Count': 'sum'})
	agg_pred_data.rename(columns={'Case Count': 'daily_confirm'}, inplace=True)

	return agg_pred_data

def plot_backfilled_actual( agg_pred_data, nowcasted_data, days):
	'''
	Function to visualize the backfilled/nowcast data against the actual data.
	'''
	fig = plt.figure(figsize=(6,2), dpi = 150)
	plt.plot(agg_pred_data.index[-(days+10):], agg_pred_data['daily_confirm'][-(days+10):], marker='.', color="indianred", markersize=6, lw=1, label="Actual", alpha=0.85)
	plt.plot(nowcasted_data.index[-(days+10):], nowcasted_data['daily_confirm'][-(days+10):], marker='.', color="teal", markersize=6, lw=1, ls='dashed', label="Nowcasted", alpha=0.85)
	plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %d, %y"))
	plt.xlabel("Date")
	plt.ylabel("Infections")
	plt.xticks(fontsize=8)
	plt.yticks(fontsize=8)
	sns.despine()
	plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=6))
	plt.legend(prop={'size': 8}, loc = 'lower center')

def load_extract_data(main_df, start, end):
	'''
	Function to load data from specified start and end dates.
	'''
	main_df['collection_date'] = pd.to_datetime(main_df['collection_date'])
	main_df.set_index('collection_date', inplace=True)
	main_df.columns = pd.to_datetime(main_df.columns)
	subset_df = main_df.loc[:,start:end]
	return subset_df

def cut_stable_data(subset_df, thresh=0.1):
	'''
	Function to remove stable data based on a threshold value.
	'''
	subset_arr = subset_df.values
	np.seterr(divide='ignore')
	max_arr = np.nanmax(subset_arr, axis=0)
	max_arr =np.expand_dims(max_arr, axis=0)
	p = (1-(subset_arr/max_arr))
	p[p < thresh] = np.nan
	subset_df_p = pd.DataFrame(p,columns=subset_df.columns, index = subset_df.index)
	return subset_df_p

def generate_fit_data(subset_df_p):
	'''
	Function to generate all the covariates for the random forest model.
	'''
	subset_df_p = subset_df_p.stack().reset_index()
	subset_df_p["day_since_first"]=subset_df_p['collection_date']-subset_df_p['level_1']
	subset_df_p["day_since_first"] = subset_df_p["day_since_first"].dt.days
	subset_df_p["day_collection"] = subset_df_p['collection_date'].dt.dayofweek
	subset_df_p = (subset_df_p.replace([np.inf, -np.inf], np.nan).dropna())
	subset_df_p['day_since_first_quad'] = subset_df_p['day_since_first'] ** 2
	subset_df_p['day_since_first_cub'] = subset_df_p['day_since_first'] ** 3
	return subset_df_p

def fit_random_forest(subset_df, thresh):
	'''
	Function to fit the random forest model.
	'''
	subset_df_p = cut_stable_data(subset_df, thresh)
	subset_df_p = generate_fit_data(subset_df_p)
	infection_data = generate_fit_data(subset_df)
	infection_data = infection_data.rename(columns={0: 'infections'})
	subset_df_p = subset_df_p.merge(infection_data, on=['collection_date', 'level_1', 'day_since_first', 'day_collection', 'day_since_first_quad', 'day_since_first_cub'], how='left')

	final = subset_df_p.drop(["collection_date","level_1"], axis=1)
	X_train = pd.concat([pd.get_dummies(final["day_collection"]),final['day_since_first'], final['infections'], final['day_since_first_quad'], final['day_since_first_cub']], axis=1) 
	y_train = final[0]
	reg = RandomForestRegressor(n_estimators = 500, random_state = 42).fit(X_train, y_train)
	plot_explained_bar(r2_score(y_train, reg.predict(X_train)), 'train')
	return reg

def make_prediction(reg, main_df, start, end, thresh):
	'''
	Use the fitted random forest model to make predictions.
	'''
	subset_df = main_df.loc[:,start:end]
	
	subset_df_p = cut_stable_data(subset_df, thresh)
	subset_df_p = generate_fit_data(subset_df_p)
	infection_data = generate_fit_data(subset_df)
	infection_data = infection_data.rename(columns={0: 'infections'})
	subset_df_p = subset_df_p.merge(infection_data, on=['collection_date', 'level_1', 'day_since_first', 'day_collection', 'day_since_first_quad', 'day_since_first_cub'], how='left')

	final = subset_df_p.drop(["collection_date","level_1"], axis=1)
	X_train = pd.concat([pd.get_dummies(final["day_collection"]),final['day_since_first'], final['infections'], final['day_since_first_quad'], final['day_since_first_cub']], axis=1) 

	y_train = final[0]
	pred = reg.predict(X_train)
	return r2_score(y_train, reg.predict(X_train)), y_train, pred, subset_df_p

def datetime_str(date):
	'''
	Convert datetime to string format
	'''
	return f'{date.value.year}-{date.value.month:02d}-{date.value.day:02d}'