import streamlit as st
import google.generativeai as genai
from databricks import sql
import pandas as pd
import time

# --- CONFIGURAÇÕES DE PÁGINA ---
st.set_page_config(
    page_title="RetailX AI Assistant 🤖",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILIZAÇÃO CUSTOMIZADA (CSS) ---
st.markdown("""
    <style>
    .main {
        background-color: #F0F2F6;
    }
    .stSidebar {
        background-color: #2962FF;
    }
    [data-testid="stSidebar"] {
        color: white;
    }
    .stSidebar [data-testid="stMarkdownContainer"] p {
        color: white;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #2962FF;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CONSTANTES PADRÃO ---
DEFAULT_GEMINI_KEY = "AIzaSyCpPyAHACq9ok1FZSfMYHaKuHMISEgWWBs"
DB_SERVER = "dbc-6f548446-a4f7.cloud.databricks.com"
DB_HTTP_PATH = "/sql/1.0/warehouses/746620086a1dd867"
DB_TOKEN = "dapi9da38fe2a47318a47f18aa06f296415f"

SCHEMA_CONTEXT = """
TABELA: workspace.droove_dev.gold_vendas_por_categoria
COLUNAS: categoria (STRING), total_vendas (DOUBLE), data_ref (DATE)

TABELA: workspace.droove_dev.gold_performance_regional
COLUNAS: regiao (STRING), faturamento_total (DOUBLE), qtd_pedidos (INT)

TABELA: workspace.droove_dev.gold_top_clientes
COLUNAS: id_cliente (STRING), nome_cliente (STRING), valor_total (DOUBLE)
"""

# --- SIDEBAR DE CONFIGURAÇÃO ---
with st.sidebar:
    st.title("Configurações ⚙️")
    
    st.subheader("Google Gemini")
    gemini_key = st.text_input("Gemini API Key", value=DEFAULT_GEMINI_KEY, type="password")
    
    # Inicializa Gemini para listar modelos
    try:
        genai.configure(api_key=gemini_key)
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        selected_model = st.selectbox("Modelo Gemini", models, index=models.index('models/gemini-1.5-flash') if 'models/gemini-1.5-flash' in models else 0)
    except Exception as e:
        st.error(f"Erro ao carregar modelos Gemini: {e}")
        selected_model = "models/gemini-1.5-flash"

    st.divider()
    
    st.subheader("Databricks")
    db_host = st.text_input("Server Host", value=DB_SERVER)
    db_path = st.text_input("HTTP Path", value=DB_HTTP_PATH)
    db_token = st.text_input("Access Token", value=DB_TOKEN, type="password")
    
    st.divider()
    st.info("RetailX AI Assistant v1.0")

# --- FUNÇÕES DE LÓGICA ---

def run_query(query, host, path, token):
    try:
        with sql.connect(server_hostname=host, 
                         http_path=path, 
                         access_token=token) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                result = cursor.fetchall()
                if not result:
                    return pd.DataFrame()
                columns = [desc[0] for desc in cursor.description]
                return pd.DataFrame(result, columns=columns)
    except Exception as e:
        st.error(f"Erro na execução do SQL: {e}")
        return None

def ask_retailx_agent(pergunta, model_name, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        # Passo A: Gerar SQL
        prompt_sql = f"""
        Você é um Engenheiro de Dados especialista em Databricks (Spark SQL).
        Sua tarefa é converter a pergunta do usuário em uma query SQL válida.
        
        Esquema das Tabelas:
        {SCHEMA_CONTEXT}
        
        Regras:
        1. Retorne APENAS o código SQL puro, sem explicações, sem blocos de código markdown (```sql).
        2. Use sempre o nome completo das tabelas (ex: workspace.droove_dev.gold_vendas_por_categoria).
        3. Se a pergunta não puder ser respondida com as tabelas acima, retorne "ERRO: Tabelas não encontradas".
        
        Pergunta: "{pergunta}"
        """
        
        response = model.generate_content(prompt_sql)
        sql_query = response.text.strip()
        
        # Limpeza básica se a IA ignorar a regra de markdown
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        
        return sql_query
    except Exception as e:
        st.error(f"Erro ao gerar SQL com Gemini: {e}")
        return None

def interpret_results(pergunta, df, model_name, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        data_summary = df.to_string(index=False)
        
        prompt_interpret = f"""
        Você é um Analista de Negócios da RetailX.
        O usuário perguntou: "{pergunta}"
        Os dados retornados do banco de dados foram:
        {data_summary}
        
        Sua tarefa:
        Resuma os resultados em linguagem de negócios clara e profissional. 
        Destaque os pontos principais. Seja conciso.
        """
        
        response = model.generate_content(prompt_interpret)
        return response.text
    except Exception as e:
        return f"Erro ao interpretar resultados: {e}"

# --- INTERFACE PRINCIPAL ---

st.title("RetailX AI Assistant 🤖")
st.subheader("Seu Analista de Dados Inteligente")

# Histórico de Chat (Session State)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibir mensagens anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "df" in message:
            with st.expander("Ver dados brutos"):
                st.dataframe(message["df"])

# Input do Usuário
if prompt := st.chat_input("Ex: Quais são as top 5 categorias em vendas?"):
    # Adiciona pergunta ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Processamento do Agente
    with st.chat_message("assistant"):
        with st.status("Pensando...", expanded=True) as status:
            # 1. Gerar SQL
            st.write("Gerando query SQL...")
            sql_query = ask_retailx_agent(prompt, selected_model, gemini_key)
            
            if sql_query and not sql_query.startswith("ERRO"):
                st.code(sql_query, language="sql")
                
                # 2. Executar no Databricks
                st.write("Executando query no Databricks...")
                df = run_query(sql_query, db_host, db_path, db_token)
                
                if df is not None:
                    if not df.empty:
                        # 3. Interpretar
                        st.write("Interpretando resultados...")
                        analise = interpret_results(prompt, df, selected_model, gemini_key)
                        status.update(label="Análise concluída!", state="complete", expanded=False)
                        
                        st.markdown(analise)
                        with st.expander("Ver dados brutos"):
                            st.dataframe(df)
                        
                        # Salva no histórico
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": analise,
                            "df": df
                        })
                    else:
                        status.update(label="Nenhum dado encontrado.", state="error", expanded=True)
                        st.warning("A consulta não retornou resultados.")
                else:
                    status.update(label="Erro na execução SQL.", state="error", expanded=True)
            else:
                status.update(label="Erro na geração da query.", state="error", expanded=True)
                st.error("Não consegui converter sua pergunta em uma query válida para as tabelas disponíveis.")
