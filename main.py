import pandas as pd

# История курса валют
data_ce = pd.read_parquet("historical_currency_exchange.parquet")
# История транзакций
data_t = pd.read_parquet("transaction_fraud_data.parquet")

# 1. Найти долю мошенических транзакций
frauded = data_t.query('is_fraud == True')
print(f"Доля мошенических транзакций: {len(frauded) / len(data_t):.2f}\n")

# 2. Топ стран по количеству мошенических транзакций
countries_frauding = frauded.groupby("country").agg('count').is_fraud.sort_values(ascending=False)
print("Топ стран по количеству мошенических операций: \n", countries_frauding, "\n")

# 3. Сколько транзакций в час в среднем совершает один клиент?
data_t['hour'] = data_t['timestamp'].dt.floor("h")  # выносим поле "час"
records_per_hour = data_t.groupby(['customer_id', 'hour']).size().reset_index(name='transactions')  # группируем по клиентам и часам
mean_per_client = records_per_hour.groupby('customer_id')['transactions'].mean()  # группируем по клиентам и аггрегируем среднее число транзакций
overall_mean = mean_per_client.mean()  # находим среднее число транзакций для всех клиентов

print(f"Среднее число транзакций клиентов в час: {overall_mean:.2f}\n")

# 4. Доля мошенических операций среди продавцов с высоким риском
risk_vendors = data_t.query('is_high_risk_vendor == True')
frauded_risk_vendors = risk_vendors.query('is_fraud == True')
print(f"Доля мошенических операций среди продавцов с высоким риском: {len(frauded_risk_vendors) / len(risk_vendors):.2f}\n")

# 5. Города и средний чек транзакции
cities = data_t.loc[:, ["city", "amount"]].groupby("city").agg('mean').amount.sort_values(ascending=False)
print("Топ городов по среднему чеку транзакции: \n", cities, "\n")

# 6. Города по убыванию средней суммы транзакций, связанных с фаст-фудом
ff_data = data_t.loc[:, ["city", "amount", "vendor_type"]].query('vendor_type == "fast_food"')
cities_ff = ff_data.loc[:, ["city", "amount"]].groupby("city").agg('mean').amount.sort_values(ascending=False)
print("Топ городов по средней сумме транзакций, связанных с фаст-фудом: \n", cities_ff, "\n")

def convert_to_usd(df):
    """
    Конвертирует любую валюту из transaction_fraud_data в USD по курсам из historical_currency_exchange
    """
    df["date"] = df["timestamp"].dt.date

    merged = pd.merge(
        df,
        data_ce,
        on='date',
        how='left'
    )

    merged['rate'] = merged.apply(
        lambda row: row[row['currency']],
        axis=1
    )

    merged['amount_usd'] = merged['amount'] / merged['rate']
    return merged[['timestamp', 'amount_usd']]   


# 7-8. Не мошеннические транзакции в долларах
not_frauded_transactions = data_t.query('is_fraud == False').loc[:, ["timestamp", "amount", "currency"]] 
not_frauded_usd = convert_to_usd(not_frauded_transactions) 

print(f"Средняя сумма транзацкии в USD среди не мошеннических операций: {not_frauded_usd.amount_usd.mean():.2f}")  # Среднее
print(f"СКО суммы транзацкии в USD среди не мошеннических операций: {not_frauded_usd.amount_usd.std():.2f}\n")

# 9-10. Мошеннические транзакции в долларах
frauded_transactions = data_t.query('is_fraud == True').loc[:, ["timestamp", "amount", "currency"]] 
frauded_usd = convert_to_usd(frauded_transactions) 

print(f"Средняя сумма транзацкии в USD среди мошеннических операций: {frauded_usd.amount_usd.mean():.2f}")
print(f"СКО суммы транзацкии в USD среди мошеннических операций: {frauded_usd.amount_usd.std():.2f}\n")

# 11. 95-ая квантиль медианного количества уникальных продавцов за последний час для каждого из клиентов
data_t["unique_merchants"] = data_t['last_hour_activity'].apply(lambda x: x['unique_merchants'])
clients = data_t.loc[:, ["customer_id", "unique_merchants"]].groupby('customer_id')['unique_merchants'].median()
quantile_95 = clients.quantile(0.95)
high_merchants_count_clients = clients[clients > quantile_95]
count_high = len(high_merchants_count_clients)  # число клиентов, выходящих за 95-ую квантиль
print("Число клиентов, демонстрирующих потенциально опасное поведение: ", count_high)
