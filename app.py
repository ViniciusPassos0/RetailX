import streamlit as st
import google.generativeai as genai
from databricks import sql
import pandas as pd
import time
import json
import os

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

# --- CARREGAR SECRETS DO STREAMLIT CLOUD ---
# Estes valores devem ser configurados em Settings > Secrets no Streamlit Cloud
try:
    GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    GEMINI_KEY = ""

try:
    DB_SERVER = st.secrets["DB_SERVER"]
except KeyError:
    DB_SERVER = ""

try:
    DB_HTTP_PATH = st.secrets["DB_HTTP_PATH"]
except KeyError:
    DB_HTTP_PATH = ""

try:
    DB_TOKEN = st.secrets["DB_TOKEN"]
except KeyError:
    DB_TOKEN = ""

DB_SCHEMA = "workspace.droove_dev"

# --- INICIALIZAR SESSION STATE ---
if "catalog" not in st.session_state:
    st.session_state.catalog = {}
if "selected_tables" not in st.session_state:
    st.session_state.selected_tables = set()
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- FUNÇÕES DE LÓGICA ---

def get_database_catalog(host, path, token, schema):
    """
    Descobre todas as tabelas no schema especificado e suas colunas/tipos.
    Retorna um dicionário com estrutura: {tabela: {coluna: tipo}}
    """
    try:
        with sql.connect(server_hostname=host, 
                         http_path=path, 
                         access_token=token) as connection:
            with connection.cursor() as cursor:
                # Listar todas as tabelas no schema
                cursor.execute(f"SHOW TABLES IN {schema}")
                tables = cursor.fetchall()
                
                catalog = {}
                for table_row in tables:
                    table_name = table_row[1]  # Nome da tabela
                    full_table_name = f"{schema}.{table_name}"
                    
                    # Obter colunas e tipos
                    cursor.execute(f"DESCRIBE TABLE {full_table_name}")
                    columns = cursor.fetchall()
                    
                    column_info = {}
                    for col in columns:
                        col_name = col[0]
                        col_type = col[1]
                        column_info[col_name] = col_type
                    
                    catalog[full_table_name] = column_info
                
                return catalog
    except Exception as e:
        st.error(f"Erro ao descobrir tabelas: {e}")
        return {}

def format_catalog_for_prompt(catalog, selected_tables):
    """
    Formata o catálogo para usar no prompt do Gemini.
    Inclui apenas as tabelas selecionadas.
    """
    schema_context = ""
    for table_name in sorted(selected_tables):
        if table_name in catalog:
            columns = catalog[table_name]
            col_str = ", ".join([f"{col} ({dtype})" for col, dtype in columns.items()])
            schema_context += f"TABELA: {table_name}\nCOLUNAS: {col_str}\n\n"
    return schema_context

def run_query(query, host, path, token):
    """Executa uma query no Databricks e retorna um DataFrame."""
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
        return None, str(e)

def ask_retailx_agent(pergunta, model_name, api_key, schema_context):
    """Gera SQL usando Gemini com base no catálogo selecionado."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        prompt_sql = f"""
        Você é um Engenheiro de Dados especialista em Databricks (Spark SQL).
        Sua tarefa é converter a pergunta do usuário em uma query SQL válida.
        
        Esquema das Tabelas Disponíveis:
        {schema_context}
        
        Regras IMPORTANTES:
        1. Retorne APENAS o código SQL puro, sem explicações, sem blocos de código markdown (```sql).
        2. Use sempre o nome completo das tabelas (ex: workspace.droove_dev.gold_vendas_por_categoria).
        3. Se a pergunta não puder ser respondida com as tabelas acima, retorne "ERRO: Tabelas não encontradas".
        4. Prefira usar tabelas de camadas mais detalhadas (silver, fact) quando disponível.
        
        Pergunta: "{pergunta}"
        """
        
        response = model.generate_content(prompt_sql)
        sql_query = response.text.strip()
        
        # Limpeza básica se a IA ignorar a regra de markdown
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        
        return sql_query
    except Exception as e:
        return None

def suggest_alternative_question(pergunta, model_name, api_key):
    """Sugere uma pergunta alternativa quando a query falha."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        prompt = f"""
        O usuário perguntou: "{pergunta}"
        Porém, essa pergunta não pode ser respondida com os dados disponíveis.
        
        Sugira uma pergunta SIMILAR que seja respondível com dados de vendas, clientes, regiões e performance.
        Retorne APENAS a pergunta sugerida, sem explicações.
        """
        
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return None

def interpret_results(pergunta, df, model_name, api_key):
    """Interpreta os resultados em linguagem de negócios."""
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

# --- SIDEBAR DE CONFIGURAÇÃO ---
with st.sidebar:
    st.title("Configurações ⚙️")
    
    st.subheader("Google Gemini")
    
    # Se houver chave nos secrets, usar; senão, mostrar input
    if GEMINI_KEY:
        st.success("✅ Gemini API Key carregada dos secrets")
        gemini_key = GEMINI_KEY
    else:
        gemini_key = st.text_input("Gemini API Key", type="password", help="Configure em Settings > Secrets no Streamlit Cloud")
        if not gemini_key:
            st.warning("⚠️ Configure GEMINI_API_KEY nos secrets do Streamlit Cloud")
    
    # Inicializa Gemini para listar modelos
    if gemini_key:
        try:
            genai.configure(api_key=gemini_key)
            models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            selected_model = st.selectbox("Modelo Gemini", models, index=models.index('models/gemini-2.5-flash') if 'models/gemini-2.5-flash' in models else 0)
        except Exception as e:
            st.error(f"Erro ao carregar modelos: {e}")
            selected_model = "models/gemini-2.5-flash"
    else:
        selected_model = "models/gemini-2.5-flash"

    st.divider()
    
    st.subheader("Databricks")
    
    # Se houver credenciais nos secrets, usar; senão, mostrar inputs
    if DB_SERVER and DB_HTTP_PATH and DB_TOKEN:
        st.success("✅ Databricks configurado dos secrets")
        db_host = DB_SERVER
        db_path = DB_HTTP_PATH
        db_token = DB_TOKEN
    else:
        st.warning("⚠️ Configure Databricks nos secrets do Streamlit Cloud")
        db_host = st.text_input("Server Host", help="Configure DB_SERVER nos secrets")
        db_path = st.text_input("HTTP Path", help="Configure DB_HTTP_PATH nos secrets")
        db_token = st.text_input("Access Token", type="password", help="Configure DB_TOKEN nos secrets")
    
    st.divider()
    
    # Botão para carregar catálogo
    if st.button("🔄 Carregar Catalog", use_container_width=True):
        if not db_host or not db_path or not db_token:
            st.error("❌ Configure as credenciais do Databricks nos secrets!")
        else:
            with st.spinner("Descobrindo tabelas no Databricks..."):
                st.session_state.catalog = get_database_catalog(db_host, db_path, db_token, DB_SCHEMA)
                if st.session_state.catalog:
                    st.session_state.selected_tables = set(st.session_state.catalog.keys())
                    st.success(f"✅ {len(st.session_state.catalog)} tabelas encontradas!")
                else:
                    st.warning("Nenhuma tabela encontrada.")
    
    # Seletor de tabelas (Catalog)
    if st.session_state.catalog:
        st.divider()
        st.subheader("📊 Catalog de Tabelas")
        
        # Agrupar tabelas por tipo (gold, silver, dim, fact)
        table_types = {}
        for table_name in sorted(st.session_state.catalog.keys()):
            table_short = table_name.split(".")[-1]
            if "gold" in table_short:
                tipo = "🥇 Gold"
            elif "silver" in table_short:
                tipo = "🥈 Silver"
            elif "dim_" in table_short:
                tipo = "📐 Dimensão (Dim)"
            elif "fact_" in table_short:
                tipo = "📊 Fato (Fact)"
            else:
                tipo = "📋 Outras"
            
            if tipo not in table_types:
                table_types[tipo] = []
            table_types[tipo].append(table_name)
        
        # Mostrar tabelas agrupadas
        for tipo in sorted(table_types.keys()):
            with st.expander(tipo, expanded=True):
                for table_name in table_types[tipo]:
                    table_short = table_name.split(".")[-1]
                    is_selected = table_name in st.session_state.selected_tables
                    
                    if st.checkbox(table_short, value=is_selected, key=table_name):
                        st.session_state.selected_tables.add(table_name)
                    else:
                        st.session_state.selected_tables.discard(table_name)
    
    st.divider()
    st.info(f"RetailX AI Assistant v2.2\n\n📊 Tabelas selecionadas: {len(st.session_state.selected_tables)}")

# --- INTERFACE PRINCIPAL ---

st.title("RetailX AI Assistant 🤖")
st.subheader("Seu Analista de Dados Inteligente")

# Verificar se há tabelas selecionadas
if not st.session_state.selected_tables:
    st.warning("⚠️ Nenhuma tabela selecionada. Clique em '🔄 Carregar Catalog' na sidebar para começar.")
else:
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
                # Verificar se gemini_key está configurada
                if not gemini_key:
                    st.error("❌ Configure a Gemini API Key nos secrets!")
                    status.update(label="Erro de configuração", state="error", expanded=True)
                else:
                    # Preparar contexto do schema com tabelas selecionadas
                    schema_context = format_catalog_for_prompt(st.session_state.catalog, st.session_state.selected_tables)
                    
                    # 1. Gerar SQL
                    st.write("Gerando query SQL...")
                    sql_query = ask_retailx_agent(prompt, selected_model, gemini_key, schema_context)
                    
                    if sql_query and not sql_query.startswith("ERRO"):
                        st.code(sql_query, language="sql")
                        
                        # 2. Executar no Databricks
                        st.write("Executando query no Databricks...")
                        result = run_query(sql_query, db_host, db_path, db_token)
                        
                        if isinstance(result, tuple):
                            # Erro na execução
                            df = None
                            error_msg = result[1]
                        else:
                            df = result
                            error_msg = None
                        
                        if df is not None and not df.empty:
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
                            # Fallback: Sugerir pergunta alternativa
                            st.write("Tentando sugerir uma pergunta alternativa...")
                            alternative = suggest_alternative_question(prompt, selected_model, gemini_key)
                            
                            if alternative:
                                status.update(label="Pergunta ajustada", state="warning", expanded=False)
                                st.warning(f"💡 **Sugestão:** {alternative}")
                                
                                # Tentar novamente com a pergunta sugerida
                                st.write("Tentando com a pergunta sugerida...")
                                sql_query_alt = ask_retailx_agent(alternative, selected_model, gemini_key, schema_context)
                                
                                if sql_query_alt and not sql_query_alt.startswith("ERRO"):
                                    st.code(sql_query_alt, language="sql")
                                    df_alt = run_query(sql_query_alt, db_host, db_path, db_token)
                                    
                                    if isinstance(df_alt, tuple):
                                        df_alt = None
                                    
                                    if df_alt is not None and not df_alt.empty:
                                        analise_alt = interpret_results(alternative, df_alt, selected_model, gemini_key)
                                        status.update(label="Análise concluída!", state="complete", expanded=False)
                                        
                                        st.markdown(analise_alt)
                                        with st.expander("Ver dados brutos"):
                                            st.dataframe(df_alt)
                                        
                                        st.session_state.messages.append({
                                            "role": "assistant", 
                                            "content": analise_alt,
                                            "df": df_alt
                                        })
                                    else:
                                        status.update(label="Erro na execução", state="error", expanded=True)
                                        st.error("Não consegui executar nem a pergunta alternativa.")
                            else:
                                status.update(label="Erro na geração da query", state="error", expanded=True)
                                st.error("Não consegui converter sua pergunta em uma query válida.")
                    else:
                        status.update(label="Erro na geração da query", state="error", expanded=True)
                        st.error("Não consegui converter sua pergunta em uma query válida para as tabelas disponíveis.")
