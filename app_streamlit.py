import streamlit as st 
import requests
import seaborn as sns
import matplotlib.pyplot as plt
from model_utils import TODOS_SINTOMAS

st.title("Sistema de Triagem Hospitalar com IA")

# MENU
pagina = st.selectbox("Escolha a tela", ["Triagem", "Correção Médica", "Admin","Dashboard"])


# TELA 1 - TRIAGEM

if pagina == "Triagem":

    st.header("Triagem do Paciente")

    cpf = st.text_input("CPF (somente números)", max_chars=11)

    sexo = st.selectbox("Sexo", ["Male", "Female"])

    idade = st.selectbox(
        "Faixa de idade",
        ["Bebe", "Crianca", "Adolescente", "Adulto", "Idoso"]
    )

    # lista completa sem redundância textual
    try:
        resposta_sintomas = requests.get("http://127.0.0.1:8000/sintomas")

        if resposta_sintomas.status_code == 200:
            queixas_modelo = resposta_sintomas.json()
        else:
            queixas_modelo = sorted(list(TODOS_SINTOMAS))

    except:
        queixas_modelo = sorted(list(TODOS_SINTOMAS))

    # selectbox do streamlit já tem busca automática
    queixa = st.selectbox(
        "Queixa principal",
        queixas_modelo,
        accept_new_options=True
    )

    sbp = st.number_input("Pressão Sistólica (SBP)", min_value=50, max_value=250, value=120)
    dbp = st.number_input("Pressão Diastólica (DBP)", min_value=30, max_value=150, value=80)
    hr = st.number_input("Frequência Cardíaca (HR)", min_value=30, max_value=220, value=80)
    rr = st.number_input("Frequência Respiratória (RR)", min_value=5, max_value=60, value=18)
    bt = st.number_input("Temperatura (BT)", min_value=30.0, max_value=43.0, value=36.5)
    sat = st.number_input("Saturação (%)", min_value=50, max_value=100, value=98)
    dor = st.number_input("Nível de dor (0-10)", min_value=0, max_value=10, value=0)

    if st.button("Classificar paciente"):

        if len(cpf) != 11 or not cpf.isdigit():
            st.error("CPF inválido! Digite 11 números.")
        
        else:
            dados = {
                "cpf": cpf,
                "Sex": sexo,
                "Age_Group": idade,
                "Chief_complain": queixa,
                "SBP": sbp,
                "DBP": dbp,
                "HR": hr,
                "RR": rr,
                "BT": bt,
                "Saturation": sat,
                "NRS_pain": dor
            }

            resposta = requests.post(
                "http://127.0.0.1:8000/predict",
                json=dados
            )

            resultado = resposta.json()
            classe = resultado["classe_predita"]

            st.subheader(f"Nível de Triagem: {classe}")

            if classe == 1:
                st.error("🔴 Emergência")
            elif classe == 2:
                st.warning("🟠 Muito urgente")
            elif classe == 3:
                st.warning("🟡 Urgente")
            elif classe == 4:
                st.success("🟢 Pouco urgente")
            else:
                st.info("🔵 Não urgente")


# TELA 2 - CORREÇÃO MÉDICA

elif pagina == "Correção Médica":

    st.header("Correção de Classificação")

    st.write("Informe o CPF do paciente e a classificação real")

    cpf_correcao = st.text_input("CPF do paciente", max_chars=11)

    classe_real = st.selectbox(
        "Classificação real",
        [1, 2, 3, 4, 5]
    )

    grupos_semanticos = [
        "Sintomas Neurologicos Agudos",
        "Dor Toracica/Cardiovascular",
        "Sintomas Respiratorios",
        "Sintomas Gastrointestinais",
        "Trauma",
        "Hemorragia",
        "Sintomas Infecciosos",
        "Sintomas Urologicos",
        "Sintomas Psiquiatricos",
        "Sintomas Dermatologicos",
        "Outros Clinicos"
    ]

    if cpf_correcao and len(cpf_correcao) == 11 and cpf_correcao.isdigit():

        try:
            resposta_dados = requests.get("http://127.0.0.1:8000/dados")

            if resposta_dados.status_code == 200:

                import pandas as pd
                df_dados = pd.DataFrame(resposta_dados.json())

                if not df_dados.empty:

                    df_paciente = df_dados[df_dados["cpf"] == cpf_correcao]

                    if not df_paciente.empty:

                        ultimo = df_paciente.iloc[-1]

                        if bool(ultimo.get("sintoma_desconhecido", False)):

                            entrada = ultimo["entrada"]

                            sintoma = entrada.get("Chief_complain", "Não informado")
                            grupo_atual = entrada.get("Semantic_Group", "Outros Clinicos")

                            st.warning("Sintoma não listado detectado. Revise o grupo semântico caso a sugestão esteja incorreta.")

                            st.info(f"Sintoma digitado: {sintoma}")
                            st.info(f"Grupo sugerido pelo sistema: {grupo_atual}")

                            novo_grupo = st.selectbox(
                                "Grupo semântico correto",
                                grupos_semanticos,
                                index=grupos_semanticos.index(grupo_atual)
                                if grupo_atual in grupos_semanticos else 10
                            )

                            if st.button("Salvar correção do grupo semântico"):

                                resposta_grupo = requests.post(
                                    "http://127.0.0.1:8000/corrigir_grupo",
                                    json={
                                        "cpf": cpf_correcao,
                                        "grupo_semantico": novo_grupo
                                    }
                                )

                                resultado_grupo = resposta_grupo.json()

                                if "erro" in resultado_grupo:
                                    st.error(resultado_grupo["erro"])
                                else:
                                    st.success("Grupo semântico atualizado com sucesso!")

        except:
            st.warning("Não foi possível carregar os dados do paciente para curadoria.")

    if st.button("Salvar correção"):

        if len(cpf_correcao) != 11 or not cpf_correcao.isdigit():
            st.error("CPF inválido!")
        
        else:
            dados = {
                "cpf": cpf_correcao,
                "classe_real": classe_real
            }

            resposta = requests.post(
                "http://127.0.0.1:8000/corrigir",
                json=dados
            )

            resultado = resposta.json()

            if "erro" in resultado:
                st.error(resultado["erro"])
            else:
                st.success("Correção salva com sucesso!")

                # feedback visual
                if classe_real == 1:
                    st.error("🔴 Emergência")
                elif classe_real == 2:
                    st.warning("🟠 Muito urgente")
                elif classe_real == 3:
                    st.warning("🟡 Urgente")
                elif classe_real == 4:
                    st.success("🟢 Pouco urgente")
                else:
                    st.info("🔵 Não urgente")

# TELA 3 - ADMIN (RETRAIN)

elif pagina == "Admin":

    st.header("Administração do Sistema")

    st.write("Re-treinar o modelo com base nos dados corrigidos pelos médicos")

    if st.button("Re-treinar modelo"):

        with st.spinner("Re-treinando modelo... "):

            try:
                resposta = requests.post("http://127.0.0.1:8000/retrain")

                if resposta.status_code == 200:
                    resultado = resposta.json()

                    if "erro" in resultado:
                        st.error(resultado["erro"])
                    else:
                        st.success("Modelo re-treinado com sucesso! ")
                else:
                    st.error("Erro ao conectar com a API")

            except:
                st.error("Erro: API não está rodando")


elif pagina == "Dashboard":

    st.header("Evolução do Modelo")

    try:
        resposta = requests.get("http://127.0.0.1:8000/dados")

        if resposta.status_code != 200:
            st.error("Erro ao buscar dados")
        else:
            dados = resposta.json()

            import pandas as pd

            df = pd.DataFrame(dados)

            if df.empty:
                st.warning("Nenhum dado ainda")
            else:

                df_corrigidos = df[df["corrigido"] == True]

                if df_corrigidos.empty:
                    st.warning("Sem dados corrigidos ainda")
                else:


                    # ACURÁCIA POR VERSÃO

                    evolucao = []

                    versoes = sorted(df_corrigidos["modelo_versao"].unique())

                    for v in versoes:
                        df_v = df_corrigidos[df_corrigidos["modelo_versao"] == v]

                        acertos = (df_v["classe_predita"] == df_v["classe_real"]).sum()
                        total = len(df_v)

                        acuracia = acertos / total if total > 0 else 0

                        evolucao.append({
                            "versao": v,
                            "acuracia": acuracia
                        })

                    df_evolucao = pd.DataFrame(evolucao)

                    st.subheader("Evolução da Acurácia")
                    st.line_chart(df_evolucao.set_index("versao"))


                    # MATRIZ DE CONFUSÃO 

                    st.subheader("Matriz de Confusão - Modelo vs Médico")

    
                    matriz = pd.crosstab(
                        df_corrigidos["classe_real"],
                        df_corrigidos["classe_predita"]
                    )


                    matriz.index = matriz.index.astype(int)
                    matriz.columns = matriz.columns.astype(int)

                    import seaborn as sns
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots()

                    sns.heatmap(
                        matriz,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        linewidths=0.5,
                        linecolor='gray',
                        cbar=True,
                        annot_kws={"size": 12},
                        ax=ax
                    )

                    ax.set_xlabel("Classe Predita (Modelo)")
                    ax.set_ylabel("Classe Real (Médico)")
                    ax.set_title("Matriz de Confusão - Classificação Modelo vs Avaliação Médica", fontsize=12)

                    st.pyplot(fig)


  
                    # FILTRO POR CASO ESPECÍFICO 
  
                    st.subheader("Análise por tipo de caso")

                    queixas = df_corrigidos["entrada"].apply(lambda x: x.get("Chief_complain", "OUTROS"))

                    df_corrigidos["queixa"] = queixas

                    filtro = st.selectbox("Escolha a queixa", df_corrigidos["queixa"].unique())

                    df_filtro = df_corrigidos[df_corrigidos["queixa"] == filtro]

                    if not df_filtro.empty:

                        evolucao_filtro = []

                        for v in versoes:
                            df_v = df_filtro[df_filtro["modelo_versao"] == v]

                            if len(df_v) == 0:
                                continue

                            acertos = (df_v["classe_predita"] == df_v["classe_real"]).sum()
                            total = len(df_v)

                            evolucao_filtro.append({
                                "versao": v,
                                "acuracia": acertos / total
                            })

                        df_filtro_chart = pd.DataFrame(evolucao_filtro)

                        if not df_filtro_chart.empty:
                            st.line_chart(df_filtro_chart.set_index("versao"))
                        else:
                            st.info("Sem dados suficientes para essa queixa")

    except:
        st.error("Erro ao conectar com a API")
