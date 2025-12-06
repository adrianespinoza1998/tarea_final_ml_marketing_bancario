# Proyecto Integrado de Machine Learning para Marketing Bancario

## 📋 Descripción del Proyecto

Este proyecto implementa un análisis completo de machine learning para optimizar campañas de marketing del Banco de Portugal, enfocadas en promover depósitos a plazo. Utilizando el dataset **Bank Marketing** de UCI, se desarrolló un pipeline end-to-end que incluye análisis exploratorio, preprocesamiento, segmentación de clientes y modelamiento predictivo avanzado.

### Contexto

El Banco de Portugal ejecuta campañas telefónicas masivas para promover depósitos a plazo. Sin embargo, contactar a toda la base de clientes sin segmentación genera:
- **Altos costos operativos** por contactos improductivos
- **Baja tasa de conversión** (~11% según el dataset)
- **Desgaste de la relación** con clientes por contactos excesivos

Este proyecto proporciona una solución basada en datos para:
1. **Segmentar clientes** en grupos con diferentes propensiones a contratar
2. **Predecir probabilidad de conversión** para priorizar contactos
3. **Optimizar recursos** enfocando esfuerzos en clientes con mayor potencial

---

## 👥 Integrantes del Equipo

- **Adrian Espinoza**
- **Dante Aguirre**
- **Sofia Alanis**

*Universidad del Desarrollo – Facultad de Ingeniería*  
*Curso: Machine Learning*

---

## 📊 Dataset

**Fuente:** [UCI Machine Learning Repository - Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)

- **Registros:** 45,211 clientes
- **Variables:** 17 features + 1 target (contratación del depósito)
- **Tipo:** Clasificación binaria (desbalanceada)
- **Variables incluyen:**
  - Demográficas: edad, trabajo, estado civil, educación
  - Financieras: balance, préstamos, historial crediticio
  - Campaña: contactos, duración llamadas, resultado previo
  - Temporales: mes, día de la semana

---

## 🚀 Instalación y Configuración

### Prerrequisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Jupyter Notebook o JupyterLab

### Pasos de Instalación

1. **Clonar el repositorio**
   ```bash
   git clone <url-del-repositorio>
   cd tarea_final_ml
   ```

2. **Crear entorno virtual (recomendado)**
   ```bash
   python -m venv venv
   
   # En macOS/Linux:
   source venv/bin/activate
   
   # En Windows:
   venv\Scripts\activate
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verificar instalación**
   ```bash
   python -c "import sklearn, xgboost, pandas, seaborn; print('✓ Todas las librerías instaladas correctamente')"
   ```

### Estructura del Proyecto

```
tarea_final_ml/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_eda_analysis.ipynb          # Análisis Exploratorio de Datos
│   ├── 02_preprocessing.ipynb         # Preprocesamiento y Feature Engineering
│   └── 03_ml_model.ipynb             # Modelamiento y Evaluación
├── data/
│   └── clean/
│       └── bank_marketing_clean.csv   # Dataset preprocesado
├── figures/
│   ├── 01_eda_analysis/              # Gráficos del EDA (15 figuras)
│   ├── 02_preprocessing/             # (sin gráficos)
│   └── 03_ml_model/                  # Gráficos de modelos (13 figuras)
└── assets/
    └── enunciado_proyecto_completo.md
```

### Ejecutar el Proyecto

**Opción 1: Ejecutar notebooks en orden**
```bash
jupyter notebook
```
Luego abrir y ejecutar en orden:
1. `notebooks/01_eda_analysis.ipynb`
2. `notebooks/02_preprocessing.ipynb`
3. `notebooks/03_ml_model.ipynb`

**Opción 2: Usar JupyterLab**
```bash
jupyter lab
```

---

## 🔍 Metodología

### 1. Análisis Exploratorio de Datos (EDA)

- **15 visualizaciones** generadas y guardadas automáticamente
- Análisis de distribuciones, correlaciones y outliers
- Identificación de patrones de conversión por segmentos
- Detección de desbalanceo de clases (88% no contrató vs 12% contrató)

**Hallazgos clave:**
- Variables con mayor correlación con conversión: `duration`, `poutcome`, `balance`
- Presencia de outliers en `balance`, `duration` y `campaign`
- Alto porcentaje de valores "unknown" en algunas variables categóricas

### 2. Preprocesamiento y Feature Engineering

**Transformaciones aplicadas:**
- **Tratamiento de outliers:** Winsorización (percentiles 1-99) en variables críticas
- **Feature Engineering:** Creación de 7 variables derivadas
  - Indicadores de "unknown" (4 variables)
  - Variables de negocio: `is_contacted_before`, `previous_exit`, `positive_balance`
- **Encoding:** One-Hot Encoding de 10 variables categóricas
- **Target:** Conversión a binario (0/1)

**Resultado:** Dataset de 45,211 × ~75 variables listo para modelamiento

### 3. Métodos No Supervisados

**PCA (Análisis de Componentes Principales)**
- 10 componentes principales analizados
- Visualización de varianza acumulada
- Herramienta exploratoria para entender dimensionalidad

**K-Means Clustering**
- **5 clusters óptimos** identificados (método del codo + silhouette)
- Métricas de validación:
  - Silhouette Score: evaluación de separación
  - Davies-Bouldin Index: medida de solapamiento
  - Inertia: compacidad de clusters
- **Segmentación de clientes** con tasas de conversión diferenciadas
- Visualización en 2D mediante proyección PCA

### 4. Modelamiento Predictivo

**Modelos implementados:**
1. ✅ **Baseline:** Regresión Logística (`class_weight='balanced'`)
2. ✅ **SVM:** Support Vector Machine con kernels RBF y lineal
3. ✅ **Random Forest:** Ensamble de árboles de decisión
4. ✅ **Gradient Boosting:** Boosting secuencial de árboles
5. ✅ **XGBoost:** Gradient boosting optimizado

**Optimización:**
- `GridSearchCV` con validación cruzada (cv=3)
- Métrica de optimización: `roc_auc`
- Búsqueda exhaustiva de hiperparámetros para cada modelo

**Evaluación:**
- División de datos: 67% entrenamiento, 33% test
- Métricas calculadas: AUC, F1-Score, Accuracy, Precision, Recall, Specificity
- 5 matrices de confusión generadas (una por modelo)
- Curvas ROC comparativas

---

## 📈 Resultados Clave

### Desempeño de Modelos

| Modelo               | AUC    | F1-Score | Accuracy | Precision | Recall | Specificity |
|---------------------|--------|----------|----------|-----------|--------|-------------|
| **XGBoost** 🏆       | 0.9388 | 0.6428   | 0.9067   | 0.7151    | 0.5825 | 0.9492      |
| Gradient Boosting   | 0.9376 | 0.6384   | 0.9057   | 0.7098    | 0.5794 | 0.9482      |
| Random Forest       | 0.9281 | 0.5850   | 0.8979   | 0.6439    | 0.5360 | 0.9427      |
| SVM                 | 0.9111 | 0.5244   | 0.8887   | 0.5566    | 0.4957 | 0.9368      |
| Baseline (LogReg)   | 0.9094 | 0.5212   | 0.8881   | 0.5540    | 0.4930 | 0.9363      |

**🏆 Modelo Ganador: XGBoost**
- **AUC: 0.9388** (excelente capacidad discriminativa)
- **Alta Precision (71.51%)**: minimiza falsos positivos
- **Alta Specificity (94.92%)**: identifica correctamente clientes que NO contratarán
- **Trade-off controlado**: balancea precisión vs cobertura

### Importancia de Variables (Top 5)

**Gradient Boosting & XGBoost coinciden:**
1. **duration** (duración de la llamada): 15-20% de importancia
2. **balance** (saldo en cuenta): 8-12%
3. **age** (edad del cliente): 6-8%
4. **pdays** (días desde último contacto): 5-7%
5. **campaign** (número de contactos): 4-6%

⚠️ **Nota sobre 'duration':** Esta variable solo se conoce DESPUÉS de la llamada, por lo que no puede usarse para predicciones pre-contacto. Para uso en producción, debe excluirse o usar promedios históricos.

### Segmentación de Clientes (K-Means)

**5 segmentos identificados con tasas de conversión diferenciadas:**

- **Cluster 0:** Tasa de conversión alta (~20-25%)
  - Perfil: Balance positivo, duración de llamada prolongada, contacto previo exitoso
  - **Acción:** Priorizar estos clientes en campañas

- **Cluster 1-2:** Tasa de conversión media (~10-15%)
  - Perfil: Mixto, con algunas características favorables
  - **Acción:** Estrategias personalizadas según características

- **Cluster 3-4:** Tasa de conversión baja (~5-8%)
  - Perfil: Sin historial previo, múltiples contactos sin éxito
  - **Acción:** Evitar contacto frecuente, considerar otros canales

### Impacto de Negocio Estimado

**Escenario actual (sin modelo):**
- Tasa de conversión global: ~11%
- Contactos necesarios por conversión: ~9 llamadas

**Con modelo predictivo (top 20% probabilidad):**
- Tasa de conversión estimada: ~25-30%
- Contactos necesarios por conversión: ~3-4 llamadas
- **Reducción de ~60% en contactos improductivos**
- **Mejora de ~2.5x en eficiencia de conversión**

---

## 💡 Recomendaciones Ejecutivas

### Para la Gerencia de Marketing

1. **Implementar scoring predictivo**
   - Usar XGBoost para asignar probabilidad de conversión a cada cliente
   - Priorizar contactos en orden descendente de score
   - Establecer umbral mínimo de probabilidad (ej: 30%)

2. **Estrategias diferenciadas por cluster**
   - **Cluster Alto Potencial:** Contacto personalizado, ofertas premium
   - **Cluster Medio:** Educación financiera, incentivos moderados
   - **Cluster Bajo:** Campañas digitales de bajo costo, evitar saturación

3. **Optimizar timing de contacto**
   - Variables temporales muestran patrones estacionales
   - `pdays` indica que el momento del contacto es crítico
   - Evitar contactar clientes recientemente rechazados

4. **Métricas de monitoreo en producción**
   - Tasa de conversión por decil de score
   - ROI de campaña (costo contacto vs ingresos por conversión)
   - Monitorear data drift en variables clave

### Próximos Pasos

1. **Validación A/B Testing:** Comparar campaña tradicional vs modelo predictivo
2. **Reentrenamiento periódico:** Actualizar modelo cada 3-6 meses
3. **Feature exclusion:** Crear versión del modelo sin `duration` para predicción pre-contacto
4. **Integración CRM:** Conectar scoring con sistema de gestión de clientes
5. **Dashboard ejecutivo:** Visualización en tiempo real de métricas de campaña

---

## 📚 Tecnologías Utilizadas

- **Python 3.x**
- **Pandas & NumPy:** Manipulación de datos
- **Scikit-learn:** Modelamiento y evaluación
- **XGBoost:** Gradient boosting optimizado
- **Matplotlib & Seaborn:** Visualizaciones
- **Jupyter Notebook:** Desarrollo interactivo
- **UCI ML Repository:** Fuente de datos

---

## 📄 Licencia

Este proyecto fue desarrollado con fines académicos para el curso de Machine Learning de la Universidad del Desarrollo.

---

## 📧 Contacto

Para consultas sobre el proyecto:
- Adrian Espinoza
- Dante Aguirre  
- Sofia Alanis

*Universidad del Desarrollo – Facultad de Ingeniería*