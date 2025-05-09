import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1. Carregamento dos dados
df_alunos = pd.read_csv('alunos.csv', low_memory=False)
df_indicadores = pd.read_csv('indicadores.csv')
df_roi = pd.read_csv('roi.json.csv')

# 2. Revisão da Estrutura
print("\n📊 Estrutura dos dados:")
print("\n🔎 alunos.csv:")
print(df_alunos.info())
print("\n🔎 indicadores.csv:")
print(df_indicadores.info())
print("\n🔎 roi.json.csv:")
print(df_roi.info())

# 3. Verificação de dados faltantes
print("\n🕳️ Verificação de dados faltantes:")
print("\n📂 alunos.csv:")
print(df_alunos.isnull().sum().sort_values(ascending=False))
print("\n📂 indicadores.csv:")
print(df_indicadores.isnull().sum().sort_values(ascending=False))
print("\n📂 roi.json.csv:")
print(df_roi.isnull().sum().sort_values(ascending=False))

# 4. Padronização de colunas para merge
if 'identificadorUnico' in df_alunos.columns:
    df_alunos.rename(columns={'identificadorUnico': 'idaluno'}, inplace=True)

for df, name in zip([df_alunos, df_indicadores, df_roi], ['alunos', 'indicadores', 'roi']):
    if 'idAluno' in df.columns:
        df.rename(columns={'idAluno': 'idaluno'}, inplace=True)
    if 'idaluno' not in df.columns:
        print(f"⚠️ Coluna 'idaluno' não encontrada no arquivo {name}.csv")

# 5. Padronização de tipos
for df in [df_alunos, df_indicadores, df_roi]:
    if 'idaluno' in df.columns:
        df['idaluno'] = df['idaluno'].astype(str)

# 6. Padronização de datas
for df, col in [(df_alunos, 'data_inicio'), (df_alunos, 'data_fim'), (df_indicadores, 'data')]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# 7. Imputação de valores ausentes (exemplo com nota e status)
if 'nota' in df_alunos.columns:
    df_alunos['nota'] = df_alunos['nota'].fillna(df_alunos['nota'].mean())
if 'status' in df_alunos.columns:
    df_alunos['status'] = df_alunos['status'].fillna(df_alunos['status'].mode()[0])

# 7.1 Preenchendo valores ausentes na coluna 'interativo'
for df in [df_indicadores, df_roi]:
    if 'interativo' in df.columns:
        df['interativo'] = df['interativo'].fillna('Não informado')

# 8. Remoção de duplicatas
df_alunos.drop_duplicates(inplace=True)
df_indicadores.drop_duplicates(inplace=True)
df_roi.drop_duplicates(inplace=True)

# 9. Normalização (exemplo com nota)
if 'nota' in df_alunos.columns:
    scaler = MinMaxScaler()
    df_alunos['nota_normalizada'] = scaler.fit_transform(df_alunos[['nota']])

# 10. Avaliação inicial da qualidade
print("\n📋 Qualidade dos dados:")
print(f"Duplicatas em alunos.csv: {df_alunos.duplicated().sum()}")
print("\nValores únicos em indicadores.csv:")
print(df_indicadores.nunique())
print("\nValores únicos em roi.json.csv:")
print(df_roi.nunique())

# 11. Cruzamentos de dados (com checagem)
df_indicadores_alunos = None
if 'idaluno' in df_alunos.columns and 'idaluno' in df_indicadores.columns:
    df_indicadores_alunos = df_indicadores.merge(df_alunos, on='idaluno', how='inner')
else:
    print("⚠️ Merge entre indicadores e alunos não pôde ser feito (faltando 'idaluno').")

df_roi_indicadores = None
if 'idaluno' in df_roi.columns and 'idaluno' in df_indicadores.columns:
    df_roi_indicadores = df_roi.merge(df_indicadores, on='idaluno', how='inner')
else:
    print("⚠️ Merge entre ROI e indicadores não pôde ser feito (faltando 'idaluno').")

# 12. Análises com checagem de colunas
if df_indicadores_alunos is not None and \
   'status' in df_indicadores_alunos.columns and \
   'TOTAL_MENSAGENS_ENVIADAS' in df_indicadores_alunos.columns:
    status_mensagens = df_indicadores_alunos.groupby('status')['TOTAL_MENSAGENS_ENVIADAS'].sum()
    print("\n📨 Mensagens enviadas por status de aluno:")
    print(status_mensagens)

if df_roi_indicadores is not None and \
   'status' in df_roi_indicadores.columns and \
   'interativo' in df_roi_indicadores.columns and \
   'TOTAL_MENSAGENS_ENVIADAS' in df_roi_indicadores.columns:
    interatividade_status = df_roi_indicadores.groupby(['status', 'interativo'])['TOTAL_MENSAGENS_ENVIADAS'].sum()
    print("\n🤖 Mensagens enviadas por status e interatividade:")
    print(interatividade_status)

# 13. Salvando os dados tratados
df_alunos.to_csv('alunos_limpo.csv', index=False)
df_indicadores.to_csv('indicadores_limpo.csv', index=False)
df_roi.to_csv('roi_limpo.csv', index=False)

print("\n✅ Análise e tratamento finalizados com sucesso! Arquivos salvos:")
print("- alunos_limpo.csv")
print("- indicadores_limpo.csv")
print("- roi_limpo.csv")
