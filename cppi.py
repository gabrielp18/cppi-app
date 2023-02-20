import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
import scipy.stats
from scipy.stats import norm

def main():
    st.set_page_config(page_title='CPPI - GP', page_icon = './favicon.ico', layout = 'wide', initial_sidebar_state = 'auto')
    st.title('Aplicação para Estratégia CPPI')
    st.markdown(
        'CPPI é uma estratégia de negociação que oferece potencial de valorização de \
        um ativo de risco, ao mesmo tempo em que fornece uma garantia de capital contra o drawdown.')
    st.sidebar.title("Variáveis do Modelo CPPI")


    #@st.cache(persist=True)
    def load_data():
        ric_history = pd.read_excel('yfinance.xlsx',  index_col='Date', parse_dates=True)
        return ric_history

    index_names = pd.DataFrame({'Index Names': {'^BVSP': 'IBOVESPA', 'VALE3.SA': 'VALE3', 'PETR4.SA': 'PETR4',
                                                'ITUB4.SA': 'ITUB4', 'BBDC4.SA': 'BBDC4','B3SA3.SA': 'B3SA3',
                                                'ABEV3.SA': 'ABEV3', 'ELET3.SA': 'ELET3', 'BBAS3.SA': 'BBAS3',
                                                'RENT3.SA': 'RENT3', 'RENT3.SA': 'RENT3.SA', 'ITSA3.SA': 'ITSA3',
                                                'WEGE3.SA': 'WEGE3', 'HAPV3.SA': 'HAPV3.SA', 'RDOR3.SA': 'RDOR3',
                                                'SUZB3.SA': 'SUZB3', 'BPAC11.SA': 'BPAC11.SA', 'JBSS3.SA': 'JBSS3',
                                                'LREN3.SA': 'LREN3', 'RADL3.SA': 'RADL3.SA', 'DXCO3.SA': 'DXCO3',
                                                }})


    df_abs = load_data()

    def cppi_func(risky_r, riskfree_rate=0.13, m=3, start=10000, floor=0.8, drawdown=None, periods_per_year=52):
        # set up the CPPI parameters
        dates = risky_r.index
        n_steps = len(dates)
        account_value = start
        floor_value = start * floor
        peak = account_value

        if isinstance(risky_r, pd.Series):
            risky_r = pd.DataFrame(risky_r, columns=["R"])

        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r[:] = riskfree_rate / periods_per_year  # fast way to set all values to a number

        # set up some DataFrames for saving intermediate values
        account_history = pd.DataFrame().reindex_like(risky_r)
        risky_w_history = pd.DataFrame().reindex_like(risky_r)
        cushion_history = pd.DataFrame().reindex_like(risky_r)
        floorval_history = pd.DataFrame().reindex_like(risky_r)
        peak_history = pd.DataFrame().reindex_like(risky_r)

        for step in range(n_steps):
            if drawdown is not None:
                peak = np.maximum(peak, account_value)
                floor_value = peak * (1 - drawdown)
            cushion = (account_value - floor_value) / account_value
            risky_w = m * cushion
            risky_w = np.minimum(risky_w, 1)
            risky_w = np.maximum(risky_w, 0)
            safe_w = 1 - risky_w
            risky_alloc = account_value * risky_w
            safe_alloc = account_value * safe_w
            # recompute the new account value at the end of this step

            account_value = risky_alloc * (1 + risky_r.iloc[step]) + safe_alloc * (1 + safe_r.iloc[step])
            # save the histories for analysis and plotting
            cushion_history.iloc[step] = cushion
            risky_w_history.iloc[step] = risky_w
            account_history.iloc[step] = account_value
            floorval_history.iloc[step] = floor_value
            peak_history.iloc[step] = peak

        risky_wealth = start * (1 + risky_r).cumprod()

        backtest_result = {
            "Wealth": account_history,
            "Risky Wealth": risky_wealth,
            "Risk Budget": cushion_history,
            "Risky Allocation": risky_w_history,
            "m": m,
            "start": start,
            "risky_r": risky_r,
            "safe_r": safe_r,
            "drawdown": drawdown,
            "peak": peak_history,
            "floor": floorval_history
        }

        return backtest_result

    def plot_metrics(inp, classifier):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=inp['Wealth'].index, y=inp['Wealth'].iloc[:, 0],
                                mode='lines',
                                name='Estratégia CPPI'))
            fig.add_trace(go.Scatter(x=inp['Risky Wealth'].index, y=inp['Risky Wealth'].iloc[:, 0],
                                mode='lines',
                                name='Buy & Hold {}'.format(classifier)))
            fig.add_trace(go.Scatter(x=inp['floor'].index, y=inp['floor'].iloc[:, 0],
                                line=dict(color='white', width=4, dash='dot'), 
                                name='Piso para o investimento'))

            fig.update_layout(xaxis_title='Data',
                   yaxis_title='Patrimônio Acumulado',
                   width=1300,
                   height=650,)

            st.plotly_chart(fig, use_container_width=False)


            ## Understanding the distribution
    def annualize_rets(r, periods_per_year=52):
        compounded_growth = (1 + r).prod()
        n_periods = r.shape[0]
        return compounded_growth ** (periods_per_year / n_periods) - 1

    def annualize_vol(r, periods_per_year=52):
        return r.std() * np.sqrt(periods_per_year)

    def sharpe_ratio(r, riskfree=0.0, periods_per_year=52):
        rf_per_period = (1 + riskfree) ** (1 / periods_per_year) - 1

        excess_ret = r - rf_per_period
        ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
        ann_vol = annualize_vol(r, periods_per_year)
        return ann_ex_ret / ann_vol

    def drawdown(return_series: pd.Series):
        wealth_index = 1000 * (1 + return_series).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        return pd.DataFrame({"Wealth": wealth_index,
                             "Previous Peak": previous_peaks,
                             "Drawdown": drawdowns})

    def var_historic(r, level=5):
        if isinstance(r, pd.DataFrame):
            return r.aggregate(var_historic, level=level)
        elif isinstance(r, pd.Series):
            return -np.percentile(r, level)
        else:
            raise TypeError("Expected r to be a Series or DataFrame")

    def cvar_historic(r, level=5):
        if isinstance(r, pd.Series):
            is_beyond = (r <= -var_historic(r, level=level))
            return -r[is_beyond].mean()
        elif isinstance(r, pd.DataFrame):
            return r.aggregate(cvar_historic, level=level)
        else:
            raise TypeError("Expected r to be a Series or DataFrame")

    def var_gaussian(r, level=5, modified=False):
        z = norm.ppf(level / 100)
        if modified:  # Cornish-Fisher VaR
            # modify the Z score based on observed skewness and kurtosis
            s = scipy.stats.skew(r)
            k = scipy.stats.kurtosis(r, fisher=False)
            z = (z +
                 (z ** 2 - 1) * s / 6 +
                 (z ** 3 - 3 * z) * (k - 3) / 24 -
                 (2 * z ** 3 - 5 * z) * (s ** 2) / 36
                 )
        return -(r.mean() + z * r.std(ddof=0))

    def summary_stats(r, riskfree=0.00, periods_per_year=52):
        """
        Assumes periods per year is 52 when assuming the data is weekly. If not, change periods_per_year!
        """
        ann_r = r.aggregate(annualize_rets, periods_per_year=periods_per_year)
        ann_vol = r.aggregate(annualize_vol, periods_per_year=periods_per_year)
        ann_sr = r.aggregate(sharpe_ratio, riskfree=0, periods_per_year=periods_per_year)
        dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
        cf_var5 = r.aggregate(var_gaussian, level=5, modified=True)
        hist_cvar5 = r.aggregate(cvar_historic, level=5)
        return pd.DataFrame({
            "Returno Anualizado (%)": ann_r*100,
            "Volatilidade Anualizada (%)": ann_vol*100,
            "VaR Paramétrico (95%)": cf_var5 * 100,
            "CVaR (95%)": hist_cvar5*100,
            "Sharpe Ratio": ann_sr,
            "Máx Drawdown (%)": dd*100
        })

    #Selecting the CPPI Method
    st.sidebar.subheader("Escolha um método")
    method_name = st.sidebar.selectbox("Methods:", ("Estratégia CPPI Não-Paramétrico", "Estratégia CPPI Paramétrico"))

    if method_name == 'Estratégia CPPI Não-Paramétrico':
        st.markdown(
        'O Método Não-Paramétrico apresenta um piso fixo, o qual é setado pelo investidor.')
        classifier = st.selectbox("Escolha o Ativo", list(index_names['Index Names']))
        clas2 = index_names[index_names['Index Names'] == classifier].index[0]
        floor_set = st.slider('Escolha o Piso (%)', 0.0, 1.0, 0.5)

        df_pct = df_abs[[clas2]].pct_change().dropna()

        date_interval = st.date_input("Escolha o período de Análise", [df_pct[[clas2]].index.min(), df_pct[[clas2]].index.max()])

        start_date = date_interval[0].strftime("%Y-%m-%d")       
        end_date = date_interval[1].strftime("%Y-%m-%d")

        ind = df_pct[[clas2]][start_date:end_date]  #Getting the date based on the index name and the date interval

        st.subheader("**Análise de Risco**")
        st.markdown('*Critérios: Taxa Livre de Risco (Selic): 13.75%, Piso: {piso}%, Sem Att Drawdown*'.format(piso=floor_set*100))

        Q = cppi_func(ind, floor=floor_set)

        # Comparing the risk profile and distribution characteristics
        metric1 = Q['Risky Wealth'].pct_change().dropna()   # Risky Portfolio
        metric2 = Q['Wealth'].pct_change().dropna()  # CPPI Portfolio
        metrics_combined = pd.concat([metric1, metric2], axis=1)
        metrics_combined.columns = ['Portfólio', 'Portfólio CPPI']
        st.write(summary_stats(metrics_combined, riskfree=0.0, periods_per_year=52).T)  ###riskfree 0% !!!!!

        # if st.sidebar.checkbox('Show CPPI plot', False):
        st.subheader("**Estratégia CPPI Não-Paramétrico**")
        plot_metrics(Q, classifier=classifier)

    elif method_name == 'Estratégia CPPI Paramétrico':
        st.markdown(
        'O Método Paramétrico apresenta um piso variável com o tempo, setado através do Drawdown.')
        classifier = st.selectbox("Escolha o Ativo", list(index_names['Index Names']))
        clas2 = index_names[index_names['Index Names'] == classifier].index[0]

        df_pct = df_abs[[clas2]].pct_change().dropna()

        date_interval = st.date_input("Escolha o período de Análise", [df_pct[[clas2]].index.min(), df_pct[[clas2]].index.max()])

        start_date = date_interval[0].strftime("%Y-%m-%d")  
        end_date = date_interval[1].strftime("%Y-%m-%d")  

        ind = df_pct[[clas2]].loc[start_date: end_date]

        st.sidebar.subheader("Estratégia CPPI Paramétrico\n *Hiperparâmetros*")

        m_ratio = st.sidebar.slider("m (Exposição ao Risco)", 0.0, 10.0, step=0.5, value= 5.0, key='m_ratio')
        rf_ratio = st.sidebar.slider("Taxa Livre de Risco %", 0.0, 30.0, step=1.0, value= 10.0, key='rf_ratio')
        drawdown_ = st.sidebar.radio("Atualização por Drawdown", ('Sim', 'Não'), key='drawdown_')

        floor_ = 0.80  #needed to prvent error message
        if drawdown_ == 'Não':
            drawdown_ratio = None
            floor_ = st.sidebar.slider("Piso (%)", 0.0, 100.0, step=1.0, value= 80.0, key='floor_')
        else:
            drawdown_ratio = st.sidebar.slider("Taxa de Drawdown (%)", 0.0, 50.0, step=1.0, value= 10.0, key='drawdown')
            drawdown_ratio = drawdown_ratio/100

        R = cppi_func(ind, riskfree_rate=rf_ratio/100, m=m_ratio, start=1000, floor=floor_/100 ,drawdown=drawdown_ratio, periods_per_year=52)

        st.subheader("**Análise de Risco**")
        # Comparing the risk profile and distribution characteristics
        metric1 = R['Risky Wealth'].pct_change().dropna()   # Risky Portfolio
        metric2 = R['Wealth'].pct_change().dropna()  # CPPI Portfolio
        metrics_combined = pd.concat([metric1, metric2], axis=1)
        metrics_combined.columns = ['Portfólio', 'Portfólio CPPI']
        st.write(summary_stats(metrics_combined, riskfree=0.0, periods_per_year=52).T)

        st.subheader("**Estratégia CPPI Paramétrico**")
        plot_metrics(R, classifier=classifier)

    st.sidebar.markdown('<a href="https://www.linkedin.com/in/gabrielpires1995/">Linkedin</a>', unsafe_allow_html=True)
    st.sidebar.markdown('<a href="gabrielp1709@gmail.com">E-mail</a>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()