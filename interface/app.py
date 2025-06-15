import streamlit as st
import pandas as pd
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import warnings
import json
import re
from typing import Dict, List, Any
warnings.filterwarnings('ignore')

class PredictionChatbot:
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
        self.context = {}
        self.predefined_prompts = self._initialize_predefined_prompts()
        
    def _initialize_predefined_prompts(self) -> Dict[str, Dict[str, str]]:
        return {
            "📊 Résumé Général": {
                "prompt": "donnez un résumé général des prédictions",
                "description": "Obtenez un aperçu complet des résultats de prédiction",
                "category": "summary"
            },
            "🔋 Analyse de Génération": {
                "prompt": "analysez la génération d'énergie",
                "description": "Détails sur la production d'énergie prédite",
                "category": "generation"
            },
            "⚡ Analyse de Consommation": {
                "prompt": "analysez la consommation d'énergie",
                "description": "Informations sur la demande énergétique",
                "category": "consumption"
            },
            "📈 Efficacité du Système": {
                "prompt": "quelle est l'efficacité du système énergétique",
                "description": "Performance et équilibre énergétique",
                "category": "efficiency"
            },
            "🌍 Facteurs Environnementaux": {
                "prompt": "impact des facteurs environnementaux",
                "description": "Influence météorologique sur l'énergie",
                "category": "environmental"
            },
            "📅 Dates avec Surplus": {
                "prompt": "quelles sont les dates avec surplus",
                "description": "Jours où la génération dépasse la consommation",
                "category": "dates"
            },
            "📅 Dates avec Déficit": {
                "prompt": "quelles sont les dates avec déficit",
                "description": "Jours où la consommation dépasse la génération",
                "category": "dates"
            },
            "📊 Tendances Temporelles": {
                "prompt": "montrez les tendances des prédictions",
                "description": "Évolution temporelle des données",
                "category": "trends"
            },            "💡 Recommandations": {
                "prompt": "donnez des recommandations",
                "description": "Suggestions d'optimisation du système",
                "category": "recommendations"
            }
        }

    def get_predefined_prompts(self) -> Dict[str, Dict[str, str]]:
        return self.predefined_prompts
        
    def _initialize_patterns(self) -> Dict[str, Dict[str, Any]]:
        return {
            'summary': {
                'patterns': [
                    r'what.*overall.*results?',
                    r'give.*summary',
                    r'overall.*performance',
                    r'general.*overview',
                    r'summarize.*predictions?',
                    r'résumé|résume.*prédictions?',
                    r'donnez.*résumé',
                    r'performance.*générale'
                ],
                'intent': 'summary'
            },
            'generation': {
                'patterns': [
                    r'generation.*power?',
                    r'energy.*production',
                    r'renewable.*energy',
                    r'power.*generation',
                    r'how.*much.*generated?',
                    r'génération.*énergie',
                    r'production.*électrique',
                    r'combien.*produit',
                    r'énergie.*renouvelable'
                ],
                'intent': 'generation'
            },
            'consumption': {
                'patterns': [
                    r'consumption.*patterns?',
                    r'energy.*demand',
                    r'power.*usage',
                    r'how.*much.*consumed?',
                    r'demand.*analysis',
                    r'consommation.*énergie',
                    r'demande.*électrique',
                    r'combien.*consommé',
                    r'usage.*électrique'
                ],
                'intent': 'consumption'
            },
            'efficiency': {
                'patterns': [
                    r'efficiency.*rate?',
                    r'energy.*balance',
                    r'surplus.*deficit',
                    r'how.*efficient',
                    r'performance.*metrics',
                    r'efficacité.*énergétique',
                    r'bilan.*énergétique',
                    r'surplus.*déficit',
                    r'performance.*système'
                ],
                'intent': 'efficiency'            },
            'environmental': {
                'patterns': [
                    r'weather.*impact',
                    r'environmental.*factors?',
                    r'temperature.*effect',
                    r'wind.*influence',
                    r'climate.*impact',
                    r'météo.*impact',
                    r'facteurs.*environnementaux',
                    r'température.*effet',
                    r'vent.*influence',
                    r'climat.*impact'
                ],
                'intent': 'environmental'
            },
            'dates': {
                'patterns': [
                    r'dates.*surplus',
                    r'dates.*déficit',
                    r'quelles.*dates',
                    r'surplus.*dates',
                    r'déficit.*dates',
                    r'when.*surplus',
                    r'when.*deficit',
                    r'which.*dates',
                    r'analyse.*dates'
                ],
                'intent': 'dates_analysis'
            },
            'trends': {
                'patterns': [
                    r'trends?.*analysis',
                    r'pattern.*over.*time',
                    r'seasonal.*variations?',
                    r'time.*series.*patterns?',
                    r'what.*trends?',
                    r'tendances?.*analyse',
                    r'motifs.*temporels',
                    r'variations.*saisonnières',
                    r'quelles.*tendances',
                    r'montrez.*tendances'
                ],
                'intent': 'trends'
            },
            'recommendations': {
                'patterns': [
                    r'recommendations?',
                    r'what.*should.*do',
                    r'suggestions?',
                    r'advice.*energy',
                    r'optimize.*system',
                    r'recommandations?',
                    r'que.*faire',
                    r'conseils.*énergie',
                    r'optimiser.*système',
                    r'donnez.*recommandations'
                ],
                'intent': 'recommendations'
            },            'specific_day': {
                'patterns': [
                    r'day.*\d+',
                    r'date.*\d{4}-\d{2}-\d{2}',
                    r'specific.*day',
                    r'particular.*date',
                    r'jour.*\d+',
                    r'date.*spécifique',
                    r'jour.*particulier'
                ],
                'intent': 'specific_day'
            },
            'dates_analysis': {
                'patterns': [
                    r'dates.*génération.*supérieure',
                    r'dates.*génération.*inférieure',
                    r'quand.*génération.*plus.*grande',
                    r'quand.*génération.*plus.*petite',
                    r'surplus.*dates',
                    r'déficit.*dates',
                    r'jours.*avec.*surplus',
                    r'jours.*avec.*déficit',
                    r'dates.*surplus',
                    r'dates.*déficit',
                    r'liste.*dates.*génération',
                    r'show.*dates.*generation',
                    r'when.*generation.*higher',
                    r'when.*generation.*lower'
                ],
                'intent': 'dates_analysis'
            }        }

    def update_context(self, results_df: pd.DataFrame) -> None:
        self.context = {
            'total_days': len(results_df),
            'avg_generation': results_df['generation_predite'].mean(),
            'avg_consumption': results_df['consommation_predite'].mean(),
            'total_generation': results_df['generation_predite'].sum(),
            'total_consumption': results_df['consommation_predite'].sum(),
            'surplus_days': (results_df['generation_predite'] > results_df['consommation_predite']).sum(),
            'deficit_days': (results_df['generation_predite'] <= results_df['consommation_predite']).sum(),
            'max_generation': results_df['generation_predite'].max(),
            'min_generation': results_df['generation_predite'].min(),
            'max_consumption': results_df['consommation_predite'].max(),
            'min_consumption': results_df['consommation_predite'].min(),
            'efficiency': (results_df['generation_predite'].sum() / results_df['consommation_predite'].sum()) * 100,
            'correlation': results_df['generation_predite'].corr(results_df['consommation_predite']),
            'env_factors': [col for col in results_df.columns if col.endswith('_predite') and 
                           col not in ['generation_predite', 'consommation_predite']],
            'results_df': results_df        }

    def classify_intent(self, user_input: str) -> str:
        user_input = user_input.lower().strip()
        
        for intent_category, intent_data in self.patterns.items():
            for pattern in intent_data['patterns']:
                if re.search(pattern, user_input):
                    return intent_data['intent']
        return 'general'
    
    def generate_response(self, user_input: str) -> str:
        """Generate response based on user intent and context"""
        intent = self.classify_intent(user_input)
        
        if not self.context:
            return "🤖 Bonjour ! Je suis votre assistant IA pour l'interprétation des prédictions énergétiques. Veuillez d'abord charger des données de prédiction pour que je puisse vous aider à les analyser."
        
        response_map = {
            'summary': self._generate_summary_response,
            'generation': self._generate_generation_response,
            'consumption': self._generate_consumption_response,
            'efficiency': self._generate_efficiency_response,
            'environmental': self._generate_environmental_response,
            'trends': self._generate_trends_response,
            'recommendations': self._generate_recommendations_response,
            'specific_day': self._generate_specific_day_response,
            'dates_analysis': self._generate_dates_analysis_response,
            'general': self._generate_general_response
        }
        
        return response_map.get(intent, self._generate_general_response)(user_input)
    
    def _generate_summary_response(self, user_input: str) -> str:
        """Generate summary response"""
        ctx = self.context
        return f"""📊 **Résumé des Prédictions Énergétiques**

🔋 **Génération d'Énergie:**
- Moyenne: {ctx['avg_generation']:.2f} MW
- Total: {ctx['total_generation']:.2f} MWh
- Plage: {ctx['min_generation']:.2f} - {ctx['max_generation']:.2f} MW

⚡ **Consommation d'Énergie:**
- Moyenne: {ctx['avg_consumption']:.2f} MW
- Total: {ctx['total_consumption']:.2f} MWh
- Plage: {ctx['min_consumption']:.2f} - {ctx['max_consumption']:.2f} MW

🎯 **Performance:**
- Efficacité énergétique: {ctx['efficiency']:.1f}%
- Jours avec surplus: {ctx['surplus_days']}/{ctx['total_days']} ({ctx['surplus_days']/ctx['total_days']*100:.1f}%)
- Corrélation génération-consommation: {ctx['correlation']:.3f}

💡 **Interprétation:** {'Le système est très efficace' if ctx['efficiency'] > 100 else 'Le système nécessite une optimisation' if ctx['efficiency'] < 90 else 'Le système fonctionne correctement'}."""

    def _generate_generation_response(self, user_input: str) -> str:
        """Generate generation-focused response"""
        ctx = self.context
        df = ctx['results_df']
        
        gen_std = df['generation_predite'].std()
        gen_trend = "croissante" if df['generation_predite'].iloc[-1] > df['generation_predite'].iloc[0] else "décroissante"
        
        return f"""🔋 **Analyse de la Génération d'Énergie**

📈 **Statistiques:**
- Production moyenne: {ctx['avg_generation']:.2f} MW
- Production totale: {ctx['total_generation']:.2f} MWh
- Variabilité (écart-type): {gen_std:.2f} MW

📊 **Tendances:**
- Tendance générale: {gen_trend}
- Production maximale: {ctx['max_generation']:.2f} MW
- Production minimale: {ctx['min_generation']:.2f} MW

🎯 **Insights:**
- La production d'énergie {'varie significativement' if gen_std > ctx['avg_generation'] * 0.2 else 'est relativement stable'}
- {'Excellent potentiel de production renouvelable' if ctx['avg_generation'] > 5000 else 'Production modérée, possibilité d' + chr(39) + 'amélioration'}"""
    
    def _generate_consumption_response(self, user_input: str) -> str:
        """Generate consumption-focused response"""
        ctx = self.context
        df = ctx['results_df']
        
        cons_std = df['consommation_predite'].std()
        cons_trend = "croissante" if df['consommation_predite'].iloc[-1] > df['consommation_predite'].iloc[0] else "décroissante"
        
        return f"""⚡ **Analyse de la Consommation d'Énergie**

📊 **Profil de Consommation:**
- Consommation moyenne: {ctx['avg_consumption']:.2f} MW
- Consommation totale: {ctx['total_consumption']:.2f} MWh
- Variabilité: {cons_std:.2f} MW

📈 **Patterns:**
- Tendance: {cons_trend}
- Pic de consommation: {ctx['max_consumption']:.2f} MW
- Consommation minimale: {ctx['min_consumption']:.2f} MW

🎯 **Analyse:**
- La demande {'fluctue beaucoup' if cons_std > ctx['avg_consumption'] * 0.15 else 'est assez stable'}
- Ratio pic/moyenne: {ctx['max_consumption']/ctx['avg_consumption']:.2f}x
- {'Gestion de la demande recommandée' if ctx['max_consumption']/ctx['avg_consumption'] > 1.5 else 'Profil de consommation équilibré'}"""
    
    def _generate_efficiency_response(self, user_input: str) -> str:
        """Generate efficiency-focused response"""
        ctx = self.context
        df = ctx['results_df']
        
        daily_surplus = df['generation_predite'] - df['consommation_predite']
        avg_surplus = daily_surplus.mean()
        
        return f"""🎯 **Analyse de l'Efficacité Énergétique**

⚖️ **Bilan Énergétique:**
- Efficacité globale: {ctx['efficiency']:.1f}%
- Surplus moyen: {avg_surplus:.2f} MW
- Jours excédentaires: {ctx['surplus_days']}/{ctx['total_days']} ({ctx['surplus_days']/ctx['total_days']*100:.1f}%)

📊 **Performance:**
- Corrélation génération-consommation: {ctx['correlation']:.3f}
- {'Système très efficace ✅' if ctx['efficiency'] > 105 else '⚠️ Système déficitaire' if ctx['efficiency'] < 95 else 'Système équilibré ⚖️'}

💡 **Recommandations:**
{self._get_efficiency_recommendations(ctx['efficiency'], ctx['surplus_days']/ctx['total_days'])}"""
    
    def _generate_environmental_response(self, user_input: str) -> str:
        """Generate environmental factors response"""
        ctx = self.context
        env_factors = ctx['env_factors']
        
        if not env_factors:
            return "❌ Aucun facteur environnemental disponible dans les données actuelles."
        
        response = "🌍 **Analyse des Facteurs Environnementaux**\n\n"
        response += f"📊 **Facteurs disponibles:** {len(env_factors)} variables\n\n"
        
        factor_descriptions = {
            'temp2_max_c_predite': '🌡️ Température maximale',
            'temp2_min_c_predite': '🌡️ Température minimale',
            'temp2_ave_c_predite': '🌡️ Température moyenne',
            'suface_pressure_pa_predite': '🌪️ Pression atmosphérique',
            'wind_speed50_max_m_s_predite': '💨 Vitesse de vent maximale',
            'wind_speed50_min_m_s_predite': '💨 Vitesse de vent minimale',
            'wind_speed50_ave_m_s_predite': '💨 Vitesse de vent moyenne',
            'prectotcorr_predite': '🌧️ Précipitations'
        }
        
        for factor in env_factors:
            desc = factor_descriptions.get(factor, factor.replace('_predite', '').replace('_', ' ').title())
            response += f"• {desc}\n"
        
        response += "\n💡 **Impact sur l'énergie:**\n"
        response += "- Les facteurs météorologiques influencent directement la production d'énergie renouvelable\n"
        response += "- La température affecte la demande énergétique (chauffage/climatisation)\n"
        response += "- Le vent est crucial pour l'énergie éolienne\n"
        response += "- Les précipitations peuvent affecter l'énergie hydroélectrique"
        
        return response

    def _generate_trends_response(self, user_input: str) -> str:
        """Generate trends analysis response"""
        ctx = self.context
        df = ctx['results_df']
        
        gen_slope = np.polyfit(range(len(df)), df['generation_predite'], 1)[0]
        cons_slope = np.polyfit(range(len(df)), df['consommation_predite'], 1)[0]
        
        return f"""📈 **Analyse des Tendances Temporelles**

🔋 **Tendance de Génération:**
- Évolution: {'+' if gen_slope > 0 else '-'}{abs(gen_slope):.2f} MW/jour
- Direction: {'📈 Croissante' if gen_slope > 0 else '📉 Décroissante'}

⚡ **Tendance de Consommation:**
- Évolution: {'+' if cons_slope > 0 else '-'}{abs(cons_slope):.2f} MW/jour
- Direction: {'📈 Croissante' if cons_slope > 0 else '📉 Décroissante'}

🎯 **Insights:**
- {'⚠️ Écart grandissant entre génération et consommation' if abs(gen_slope - cons_slope) > 10 else '✅ Évolution équilibrée'}
- Prédictions couvrant {ctx['total_days']} jours
- {'Optimisation nécessaire' if gen_slope < 0 and cons_slope > 0 else 'Tendances favorables'}"""
    
    def _generate_recommendations_response(self, user_input: str) -> str:
        """Generate recommendations response"""
        ctx = self.context
        
        recommendations = []
        
        # Efficiency-based recommendations
        if ctx['efficiency'] < 90:
            recommendations.append("🔧 Augmenter la capacité de génération d'énergie renouvelable")
        elif ctx['efficiency'] > 120:
            recommendations.append("💰 Considérer la vente d'excédent énergétique")
        
        # Balance-based recommendations
        if ctx['surplus_days'] / ctx['total_days'] < 0.3:
            recommendations.append("⚡ Implémenter des stratégies de réduction de la consommation")
        
        # Correlation-based recommendations
        if abs(ctx['correlation']) < 0.3:
            recommendations.append("📊 Améliorer la synchronisation entre génération et demande")
        
        if not recommendations:
            recommendations.append("✅ Le système fonctionne de manière optimale")
            recommendations.append("🔄 Continuer la surveillance et maintenance préventive")
        
        response = "💡 **Recommandations Intelligentes**\n\n"
        for i, rec in enumerate(recommendations, 1):
            response += f"{i}. {rec}\n"
        
        response += f"\n🎯 **Priorités:**\n"
        response += "- Optimisation énergétique\n- Réduction des coûts\n- Durabilité environnementale"
        return response

    def _generate_specific_day_response(self, user_input: str) -> str:
        """Generate specific day analysis response"""
        day_match = re.search(r'(\d+)', user_input)
        if day_match:
            day_num = int(day_match.group(1))
            if 0 <= day_num < len(self.context['results_df']):
                df = self.context['results_df']
                day_data = df.iloc[day_num]
                
                return f"""📅 **Analyse du Jour {day_num + 1}**

🔋 **Génération:** {day_data['generation_predite']:.2f} MW
⚡ **Consommation:** {day_data['consommation_predite']:.2f} MW
⚖️ **Bilan:** {day_data['generation_predite'] - day_data['consommation_predite']:.2f} MW ({'Surplus' if day_data['generation_predite'] > day_data['consommation_predite'] else 'Déficit'})

📊 **Comparaison avec la moyenne:**
- Génération: {((day_data['generation_predite'] / self.context['avg_generation'] - 1) * 100):+.1f}%
- Consommation: {((day_data['consommation_predite'] / self.context['avg_consumption'] - 1) * 100):+.1f}%"""
        
        return "❌ Numéro de jour invalide. Veuillez spécifier un jour entre 1 et " + str(self.context['total_days'])
    
    def _generate_general_response(self, user_input: str) -> str:
        """Generate general response for unclassified intents"""
        suggestions = [
            "📊 'Donnez-moi un résumé des prédictions'",
            "🔋 'Comment est la génération d'énergie ?'",
            "⚡ 'Analysez la consommation'",
            "🎯 'Quelle est l'efficacité du système ?'",
            "🌍 'Quels sont les facteurs environnementaux ?'",
            "📈 'Montrez-moi les tendances'",
            "💡 'Quelles sont vos recommandations ?'"
        ]
        
        return f"""🤖 **Assistant IA pour Prédictions Énergétiques**

Je peux vous aider à analyser vos prédictions énergétiques ! Voici quelques questions que vous pouvez me poser :

{chr(10).join(suggestions)}

💬 **Ou posez-moi directement vos questions sur :**
- Performance énergétique
- Bilans et efficacité
- Facteurs environnementaux
- Tendances temporelles
- Recommandations d'optimisation"""
    
    def _get_efficiency_recommendations(self, efficiency: float, surplus_ratio: float) -> str:
        """Get specific efficiency recommendations"""
        if efficiency > 110:
            return "✅ Excellent rendement ! Envisagez la vente d'excédent ou l'expansion."
        elif efficiency > 95:
            return "🎯 Bon équilibre. Optimisez les pics de consommation."
        elif efficiency > 85:
            return "⚠️ Améliorations nécessaires. Augmentez la capacité ou réduisez la demande."
        else:
            return "🚨 Déficit critique ! Actions urgentes requises."

    def _generate_dates_analysis_response(self, user_input: str) -> str:
        """Generate response for dates analysis questions"""
        ctx = self.context
        df = ctx['results_df']
        
        # Calculer les surplus et déficits
        surplus_mask = df['generation_predite'] > df['consommation_predite']
        surplus_dates = df[surplus_mask]['date'].tolist()
        deficit_dates = df[~surplus_mask]['date'].tolist()
        
        # Analyser le type de question
        user_input_lower = user_input.lower()
        
        if any(word in user_input_lower for word in ['supérieure', 'supérieur', 'surplus', 'plus.*grande', 'higher']):
            # Questions sur les dates avec surplus
            if surplus_dates:
                dates_str = "\n".join([f"📅 {date}" for date in surplus_dates[:10]])  # Limiter à 10 dates
                more_info = f"\n\n📋 **Total:** {len(surplus_dates)} jours avec surplus" if len(surplus_dates) > 10 else f"\n\n📋 **Total:** {len(surplus_dates)} jours"
                
                return f"""🟢 **Dates où la Génération > Consommation**

{dates_str}{more_info}

📊 **Statistiques:**
- Pourcentage de jours avec surplus: {len(surplus_dates)/len(df)*100:.1f}%
- Surplus moyen: {df[surplus_mask]['generation_predite'].mean() - df[surplus_mask]['consommation_predite'].mean():.2f} MW

💡 **Interprétation:** {'Excellent équilibre énergétique' if len(surplus_dates)/len(df) > 0.6 else 'Bon potentiel d' + chr(39) + 'optimisation' if len(surplus_dates)/len(df) > 0.4 else 'Nécessite une amélioration du système'}"""
            else:
                return "🔴 **Aucune date trouvée où la génération est supérieure à la consommation.**\n\n💡 Le système fonctionne constamment en déficit énergétique."
        
        elif any(word in user_input_lower for word in ['inférieure', 'inférieur', 'déficit', 'plus.*petite', 'lower']):
            # Questions sur les dates avec déficit
            if deficit_dates:
                dates_str = "\n".join([f"📅 {date}" for date in deficit_dates[:10]])  # Limiter à 10 dates
                more_info = f"\n\n📋 **Total:** {len(deficit_dates)} jours avec déficit" if len(deficit_dates) > 10 else f"\n\n📋 **Total:** {len(deficit_dates)} jours"
                
                return f"""🔴 **Dates où la Génération < Consommation**

{dates_str}{more_info}

📊 **Statistiques:**
- Pourcentage de jours avec déficit: {len(deficit_dates)/len(df)*100:.1f}%
- Déficit moyen: {df[~surplus_mask]['consommation_predite'].mean() - df[~surplus_mask]['generation_predite'].mean():.2f} MW

💡 **Interprétation:** {'Système très déséquilibré' if len(deficit_dates)/len(df) > 0.8 else 'Déséquilibre modéré' if len(deficit_dates)/len(df) > 0.5 else 'Bon équilibre énergétique'}"""
            else:
                return "🟢 **Aucune date trouvée où la génération est inférieure à la consommation.**\n\n💡 Le système fonctionne constamment en surplus énergétique!"
        
        else:
            # Réponse générale sur l'analyse des dates
            return f"""📅 **Analyse Complète des Dates**

🟢 **Dates avec Surplus (Génération > Consommation):**
- Nombre: {len(surplus_dates)} jours ({len(surplus_dates)/len(df)*100:.1f}%)
- Premières dates: {', '.join(surplus_dates[:3]) if surplus_dates else 'Aucune'}

🔴 **Dates avec Déficit (Génération < Consommation):**
- Nombre: {len(deficit_dates)} jours ({len(deficit_dates)/len(df)*100:.1f}%)
- Premières dates: {', '.join(deficit_dates[:3]) if deficit_dates else 'Aucune'}

💡 **Questions spécifiques que vous pouvez poser:**
- "Quelles sont les dates où la génération est supérieure ?"
- "Montrez-moi les dates avec déficit énergétique"
- "Quand avons-nous un surplus d'énergie ?" """

# Initialize chatbot
@st.cache_resource
def get_chatbot():
    """Get a fresh instance of the prediction chatbot"""
    return PredictionChatbot()

st.set_page_config(page_title="🔋 Renewable Energy Forecasting System", layout="wide")
st.title("📊 Renewable Energy Forecasting System")

# Guide d'utilisateur
with st.expander("📖 Guide d'Utilisateur - Comment utiliser cette interface", expanded=False):
    st.markdown("""
    ## 🎯 **Objectif du Système**
    Cette interface permet d'analyser et de prédire la production et consommation d'énergie renouvelable 
    pour optimiser la gestion énergétique et éviter les interruptions de service.
    
    ## 📊 **Format des Données Requis**
    
    ### **📁 Structure du Fichier CSV**
    Votre fichier de données doit contenir les colonnes suivantes :
    
    #### **🔋 Colonnes Énergétiques (Obligatoires)**
    - `generation_predite` : Production d'énergie prédite (en MW)
    - `consommation_predite` : Consommation d'énergie prédite (en MW)
    - `date` : Date au format YYYY-MM-DD (ex: 2024-01-15)
    
    #### **🌍 Facteurs Environnementaux (Optionnels)**
    - `temp2_max_c_predite` : Température maximale (°C)
    - `temp2_min_c_predite` : Température minimale (°C)
    - `temp2_ave_c_predite` : Température moyenne (°C)
    - `suface_pressure_pa_predite` : Pression atmosphérique (Pa)
    - `wind_speed50_max_m_s_predite` : Vitesse du vent maximale (m/s)
    - `wind_speed50_min_m_s_predite` : Vitesse du vent minimale (m/s)
    - `wind_speed50_ave_m_s_predite` : Vitesse du vent moyenne (m/s)
    - `prectotcorr_predite` : Précipitations corrigées
    
    ### **💡 Exemple de Structure CSV**
    ```
    date,generation_predite,consommation_predite,temp2_max_c_predite,wind_speed50_ave_m_s_predite
    2024-01-01,150.5,120.3,25.4,12.8
    2024-01-02,145.2,125.7,24.1,11.5
    2024-01-03,160.8,130.2,26.2,13.2
    ```
    
    ## 🚀 **Comment Utiliser l'Interface**
    
    ### **Étape 1: Préparation des Données**
    1. 📋 Assurez-vous que votre fichier CSV contient au minimum les colonnes obligatoires
    2. 📅 Vérifiez que les dates sont au format correct (YYYY-MM-DD)
    3. 🔢 Confirmez que les valeurs numériques sont propres (pas de texte dans les colonnes numériques)
    
    ### **Étape 2: Exécution des Prédictions**
    1. 📁 Placez votre fichier CSV dans le dossier `Data/` du projet
    2. 🔧 Exécutez le notebook de prédiction LSTM pour générer les résultats
    3. 📊 Les résultats seront sauvegardés automatiquement
      ### **Étape 3: Analyse des Résultats**
    1. 📈 **Visualisations** : Graphiques interactifs de génération vs consommation
    2. 🌍 **Facteurs Environnementaux** : Impact météorologique sur l'énergie
    3. 📊 **Analyse des Surplus/Déficits** : Identification des jours critiques
    4. 🤖 **Assistant IA** : Questions prédéfinies pour une analyse approfondie
    
    ## 🤖 **Assistant IA - Chatbot Intelligent**
    
    ### **🎯 Fonctionnalités du Chatbot**
    L'Assistant IA vous permet d'obtenir des analyses détaillées et personnalisées de vos prédictions énergétiques 
    sans avoir besoin de saisir des questions complexes.
    
    ### **📋 Questions Prédéfinies Disponibles**
    
    #### **📊 Analyses Générales**
    - **📊 Résumé Général** : Vue d'ensemble complète des prédictions avec statistiques clés
    - **📈 Efficacité du Système** : Performance globale et équilibre énergétique
    - **📊 Tendances Temporelles** : Évolution des données dans le temps
    
    #### **🔋 Analyses Énergétiques**
    - **🔋 Analyse de Génération** : Détails sur la production d'énergie prédite
      - Statistiques de production (moyenne, total, variabilité)
      - Tendances de génération (croissante/décroissante)
      - Pics et creux de production
    
    - **⚡ Analyse de Consommation** : Informations sur la demande énergétique
      - Patterns de consommation
      - Variabilité de la demande
      - Comparaison avec la production
    
    #### **🌍 Facteurs Externes**
    - **🌍 Facteurs Environnementaux** : Impact météorologique sur l'énergie
      - Influence de la température sur la demande
      - Effet du vent sur la production éolienne
      - Impact des précipitations
      - Corrélations météo-énergie
    
    #### **📅 Analyses Temporelles**
    - **📅 Dates avec Surplus** : Identification des jours d'excédent énergétique
      - Liste des dates où génération > consommation
      - Magnitude des surplus
      - Recommandations pour valoriser l'excédent
    
    - **📅 Dates avec Déficit** : Identification des jours de déficit énergétique
      - Liste des dates où consommation > génération
      - Magnitude des déficits
      - Alertes et recommandations d'action
    
    #### **💡 Optimisation**
    - **💡 Recommandations** : Suggestions personnalisées d'optimisation
      - Actions pour améliorer l'efficacité
      - Stratégies de gestion des surplus/déficits
      - Optimisation des opérations
    
    ### **🚀 Comment Utiliser le Chatbot**
    
    #### **Étape 1: Sélection**
    1. 📋 Allez à la section "💬 Assistant IA pour l'Analyse Énergétique"
    2. 🎯 Choisissez une question dans la liste déroulante
    3. ℹ️ Lisez la description qui explique ce que l'analyse va révéler
    
    #### **Étape 2: Analyse**
    1. 🤖 Cliquez sur "🤖 Obtenir l'Analyse"
    2. ⏳ Attendez quelques secondes pendant le traitement
    3. 📋 Consultez la réponse détaillée dans le panneau extensible
    
    #### **Étape 3: Approfondissement**
    1. 🔗 Explorez les questions de suivi suggérées
    2. 🔄 Testez différentes analyses pour une vue complète
    3. 📊 Combinez les insights pour une stratégie optimale
    
    ### **💡 Conseils d'Utilisation**
    
    #### **🎯 Pour une Analyse Complète**
    1. **Commencez par** : "📊 Résumé Général" pour avoir une vue d'ensemble
    2. **Puis explorez** : Les analyses spécifiques (génération, consommation)
    3. **Approfondissez avec** : Les facteurs environnementaux et dates critiques
    4. **Terminez par** : Les recommandations pour l'optimisation
    
    #### **⚡ Utilisation Rapide**
    - **Pour identifier les problèmes** : "📅 Dates avec Déficit"
    - **Pour optimiser la production** : "🔋 Analyse de Génération"
    - **Pour comprendre les variations** : "🌍 Facteurs Environnementaux"
    - **Pour des actions concrètes** : "💡 Recommandations"
    
    ### **🔍 Interprétation des Réponses**
    
    #### **📊 Métriques Importantes**
    - **MW (Mégawatts)** : Puissance instantanée de génération/consommation
    - **MWh (Mégawatt-heures)** : Énergie totale sur une période
    - **Efficacité (%)** : Ratio génération/consommation × 100
    - **Corrélation** : Relation entre facteurs (-1 à +1)
    
    #### **🚨 Signaux d'Alerte**
    - **Déficits récurrents** : Risque d'interruption de service
    - **Faible efficacité (<90%)** : Système sous-optimal
    - **Forte variabilité** : Instabilité du système
    - **Corrélations négatives** : Facteurs défavorables
    
    ## ⚠️ **Points Importants**
    
    ### **✅ Bonnes Pratiques**
    - Utilisez des données historiques d'au moins 30 jours pour de meilleures prédictions
    - Assurez-vous que les données météorologiques correspondent à la région d'étude
    - Vérifiez la cohérence des unités de mesure (MW, °C, m/s, etc.)
    
    ### **❌ Erreurs Courantes à Éviter**
    - ❌ Colonnes manquantes ou mal nommées
    - ❌ Dates au mauvais format ou incohérentes
    - ❌ Valeurs négatives irréalistes pour la génération/consommation
    - ❌ Données manquantes (NaN) non traitées
    
    ## 🆘 **Support**
    Si vous rencontrez des problèmes :
    1. 🔍 Vérifiez que votre fichier CSV respecte le format requis
    2. 📋 Consultez les exemples de données dans le dossier `Data/`
    3. 🔄 Redémarrez l'interface si nécessaire
    """)

st.divider()  # Ligne de séparation

def display_prediction_plots(results_df):
    st.subheader("📈 Visualisations des Prédictions")
    
    # Convert date column to datetime if it's not already
    if 'date' in results_df.columns:
        results_df['date'] = pd.to_datetime(results_df['date'])
    
    # Identifier les colonnes de facteurs environnementaux
    env_columns = [col for col in results_df.columns if col.endswith('_predite') and 
                   col not in ['generation_predite', 'consommation_predite']]
    
    # Mapping des noms de colonnes vers des noms plus lisibles
    column_mapping = {
        'temp2_max_c_predite': '🌡️ Température Max (°C)',
        'temp2_min_c_predite': '🌡️ Température Min (°C)', 
        'temp2_ave_c_predite': '🌡️ Température Moyenne (°C)',
        'suface_pressure_pa_predite': '🌪️ Pression Atmosphérique (Pa)',
        'wind_speed50_max_m_s_predite': '💨 Vitesse Vent Max (m/s)',
        'wind_speed50_min_m_s_predite': '💨 Vitesse Vent Min (m/s)',
        'wind_speed50_ave_m_s_predite': '💨 Vitesse Vent Moyenne (m/s)',
        'prectotcorr_predite': '🌧️ Précipitations'
    }
    
    # 1. Line plot comparison: Generation vs Consumption
    st.subheader("🔋 Comparaison Génération vs Consommation")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=results_df['date'],
        y=results_df['generation_predite'],
        mode='lines',
        name='Génération Prédite',
        line=dict(color='#2E8B57', width=2),
        hovertemplate='<b>Génération</b><br>Date: %{x}<br>Valeur: %{y:.2f} MW<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=results_df['date'],
        y=results_df['consommation_predite'],
        mode='lines',
        name='Consommation Prédite',
        line=dict(color='#FF6B6B', width=2),
        hovertemplate='<b>Consommation</b><br>Date: %{x}<br>Valeur: %{y:.2f} MW<extra></extra>'
    ))
    
    fig.update_layout(
        title='📊 Évolution des Prédictions dans le Temps',
        xaxis_title='Date',
        yaxis_title='Puissance (MW)',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 2. Facteurs Environnementaux - Sélection par l'utilisateur
    if env_columns:
        st.subheader("🌍 Prédictions des Facteurs Environnementaux")
        
        # Permettre à l'utilisateur de choisir les facteurs à visualiser
        st.write("**Sélectionnez les facteurs environnementaux à visualiser :**")
        
        # Créer des colonnes pour organiser les checkboxes
        cols = st.columns(3)
        selected_factors = []
        
        for i, col in enumerate(env_columns):
            display_name = column_mapping.get(col, col.replace('_predite', '').replace('_', ' ').title())
            with cols[i % 3]:
                if st.checkbox(display_name, key=f"env_{col}", value=True):
                    selected_factors.append(col)
        
        if selected_factors:
            # Option pour choisir le type de visualisation
            viz_type = st.radio(
                "Type de visualisation :",
                ["Graphiques individuels", "Graphique combiné", "Matrice de comparaison"],
                horizontal=True
            )
            
            if viz_type == "Graphiques individuels":
                # Afficher chaque facteur dans un graphique séparé
                cols_per_row = 2
                for i in range(0, len(selected_factors), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col in enumerate(selected_factors[i:i+cols_per_row]):
                        with cols[j]:
                            fig_env = go.Figure()
                            fig_env.add_trace(go.Scatter(
                                x=results_df['date'],
                                y=results_df[col],
                                mode='lines+markers',
                                name=column_mapping.get(col, col),
                                line=dict(width=2),
                                marker=dict(size=4),
                                hovertemplate=f'<b>{column_mapping.get(col, col)}</b><br>Date: %{{x}}<br>Valeur: %{{y:.2f}}<extra></extra>'
                            ))
                            
                            fig_env.update_layout(
                                title=column_mapping.get(col, col),
                                xaxis_title='Date',
                                yaxis_title='Valeur',
                                template='plotly_white',
                                height=400,
                                showlegend=False
                            )
                            
                            st.plotly_chart(fig_env, use_container_width=True)
            
            elif viz_type == "Graphique combiné":
                # Afficher tous les facteurs sélectionnés sur un seul graphique
                fig_combined = go.Figure()
                
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
                
                for i, col in enumerate(selected_factors):
                    fig_combined.add_trace(go.Scatter(
                        x=results_df['date'],
                        y=results_df[col],
                        mode='lines',
                        name=column_mapping.get(col, col),
                        line=dict(width=2, color=colors[i % len(colors)]),
                        hovertemplate=f'<b>{column_mapping.get(col, col)}</b><br>Date: %{{x}}<br>Valeur: %{{y:.2f}}<extra></extra>'
                    ))
                
                fig_combined.update_layout(
                    title='🌍 Évolution des Facteurs Environnementaux',
                    xaxis_title='Date',
                    yaxis_title='Valeurs (échelles différentes)',
                    hovermode='x unified',
                    template='plotly_white',
                    height=600,
                    showlegend=True
                )
                
                st.plotly_chart(fig_combined, use_container_width=True)
                st.info("ℹ️ Note: Les facteurs ont des échelles différentes et sont affichés sur le même graphique pour comparaison des tendances.")
            
            elif viz_type == "Matrice de comparaison":
                # Créer une matrice de sous-graphiques
                import math
                n_factors = len(selected_factors)
                n_cols = 2
                n_rows = math.ceil(n_factors / n_cols)
                
                fig_matrix = make_subplots(
                    rows=n_rows, 
                    cols=n_cols,
                    subplot_titles=[column_mapping.get(col, col) for col in selected_factors],
                    vertical_spacing=0.08
                )
                
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
                
                for i, col in enumerate(selected_factors):
                    row = (i // n_cols) + 1
                    col_num = (i % n_cols) + 1
                    
                    fig_matrix.add_trace(
                        go.Scatter(
                            x=results_df['date'],
                            y=results_df[col],
                            mode='lines',
                            name=column_mapping.get(col, col),
                            line=dict(width=2, color=colors[i % len(colors)]),
                            showlegend=False
                        ),
                        row=row, col=col_num
                    )
                
                fig_matrix.update_layout(
                    title='🌍 Matrice de Comparaison des Facteurs Environnementaux',
                    height=400 * n_rows,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_matrix, use_container_width=True)
        
        # 3. Statistiques des facteurs environnementaux
        if selected_factors:
            st.subheader("📊 Statistiques des Facteurs Environnementaux")
            
            cols = st.columns(len(selected_factors))
            for i, col in enumerate(selected_factors):
                with cols[i]:
                    mean_val = results_df[col].mean()
                    std_val = results_df[col].std()
                    min_val = results_df[col].min()
                    max_val = results_df[col].max()
                    
                    st.metric(
                        column_mapping.get(col, col),
                        f"{mean_val:.2f}",
                        delta=f"σ: {std_val:.2f}"
                    )
                    st.caption(f"Min: {min_val:.2f} | Max: {max_val:.2f}")
    
    else:
        st.info("ℹ️ Aucun facteur environnemental trouvé dans les données. Assurez-vous que le notebook a été exécuté avec succès.")
    
    # 4. Bar chart showing surplus/deficit by day
    st.subheader("⚡ Surplus/Déficit Énergétique par Jour")
    
    results_df['surplus_deficit'] = results_df['generation_predite'] - results_df['consommation_predite']
    results_df['status'] = results_df['surplus_deficit'].apply(lambda x: 'Surplus' if x > 0 else 'Déficit')
    
    fig_bar = px.bar(
        results_df,
        x='date',
        y='surplus_deficit',
        color='status',
        color_discrete_map={'Surplus': '#27AE60', 'Déficit': '#E74C3C'},
        title='📊 Surplus/Déficit Énergétique Quotidien',
        labels={'surplus_deficit': 'Surplus/Déficit (MW)', 'date': 'Date'}
    )
    
    fig_bar.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.7)
    fig_bar.update_layout(height=400, template='plotly_white')
    
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # 3. Distribution plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Distribution de la Génération")
        fig_dist_gen = px.histogram(
            results_df,
            x='generation_predite',
            nbins=20,
            title='Distribution des Prédictions de Génération',
            color_discrete_sequence=['#2E8B57']
        )
        fig_dist_gen.update_layout(height=400, template='plotly_white')
        st.plotly_chart(fig_dist_gen, use_container_width=True)
    
    with col2:
        st.subheader("📊 Distribution de la Consommation")
        fig_dist_cons = px.histogram(
            results_df,
            x='consommation_predite',
            nbins=20,
            title='Distribution des Prédictions de Consommation',
            color_discrete_sequence=['#FF6B6B']
        )
        fig_dist_cons.update_layout(height=400, template='plotly_white')
        st.plotly_chart(fig_dist_cons, use_container_width=True)
    
    # 4. Statistics summary
    st.subheader("📋 Statistiques Récapitulatives")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_generation = results_df['generation_predite'].mean()
        st.metric("Génération Moyenne", f"{avg_generation:.2f} MW")
    
    with col2:
        avg_consumption = results_df['consommation_predite'].mean()
        st.metric("Consommation Moyenne", f"{avg_consumption:.2f} MW")
    
    with col3:
        avg_surplus = results_df['surplus_deficit'].mean()
        st.metric("Surplus Moyen", f"{avg_surplus:.2f} MW", 
                 delta=f"{avg_surplus:.2f}")
    
    with col4:
        efficiency = (results_df['generation_predite'].sum() / results_df['consommation_predite'].sum()) * 100
        st.metric("Efficacité Énergétique", f"{efficiency:.1f}%")
    
    # 5. Time series decomposition view
    st.subheader("📈 Analyse Temporelle")
    
    # Create a subplot with generation and consumption trends
    fig_trends = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Tendance de la Génération', 'Tendance de la Consommation'),
        vertical_spacing=0.1
    )
    
    # Add generation trend
    fig_trends.add_trace(
        go.Scatter(
            x=results_df['date'],
            y=results_df['generation_predite'],
            mode='lines+markers',
            name='Génération',
            line=dict(color='#2E8B57', width=2),
            marker=dict(size=4)
        ),
        row=1, col=1
    )
    
    # Add consumption trend
    fig_trends.add_trace(
        go.Scatter(
            x=results_df['date'],
            y=results_df['consommation_predite'],
            mode='lines+markers',
            name='Consommation',
            line=dict(color='#FF6B6B', width=2),
            marker=dict(size=4)
        ),
        row=2, col=1
    )
    
    fig_trends.update_layout(
        height=600,
        title_text="📊 Analyse des Tendances Temporelles",
        template='plotly_white',
        showlegend=True
    )
    
    fig_trends.update_xaxes(title_text="Date", row=2, col=1)
    fig_trends.update_yaxes(title_text="Génération (MW)", row=1, col=1)
    fig_trends.update_yaxes(title_text="Consommation (MW)", row=2, col=1)
    
    st.plotly_chart(fig_trends, use_container_width=True)
    
    # 6. Correlation analysis
    st.subheader("🔗 Analyse de Corrélation")
    
    correlation = results_df['generation_predite'].corr(results_df['consommation_predite'])
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.metric("Corrélation Génération-Consommation", f"{correlation:.3f}")
        
        # Interpretation
        if correlation > 0.7:
            st.success("🟢 Corrélation forte positive")
        elif correlation > 0.3:
            st.warning("🟡 Corrélation modérée positive")
        elif correlation > -0.3:
            st.info("🔵 Corrélation faible")
        elif correlation > -0.7:
            st.warning("🟡 Corrélation modérée négative")
        else:
            st.error("🔴 Corrélation forte négative")
    
    with col2:
        # Scatter plot of generation vs consumption
        fig_scatter = px.scatter(
            results_df,
            x='generation_predite',
            y='consommation_predite',
            title='Corrélation Génération vs Consommation',
            trendline="ols",
            labels={'generation_predite': 'Génération (MW)', 'consommation_predite': 'Consommation (MW)'}
        )
        fig_scatter.update_layout(height=400, template='plotly_white')
        st.plotly_chart(fig_scatter, use_container_width=True)

# Chemin du notebook à exécuter
notebook_path = r"c:\Users\Idea\Documents\Programming languages\Python\Notebooks\Time Series\Project_1\Notebooks\LSTM complet interface.ipynb"
results_path = r"c:\Users\Idea\Documents\Programming languages\Python\Notebooks\Time Series\Project_1\Notebooks\results.csv"
uploaded_dataset_path = r"c:\Users\Idea\Documents\Programming languages\Python\Notebooks\Time Series\Project_1\Data\uploaded_test_data.csv"

# Étape 1 : Téléversement du fichier
st.subheader("📂 Téléversez votre dataset")
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type=["csv"])

if uploaded_file is not None:
    # Sauvegarder le fichier téléversé
    with open(uploaded_dataset_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"✅ Fichier téléversé et sauvegardé à : {uploaded_dataset_path}")
    
    # Étape 2 : Exécuter le notebook
    st.info("⏳ Exécution du notebook en cours...")
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        ep.preprocess(notebook, {"metadata": {"path": os.path.dirname(notebook_path)}})
        st.success("✅ Notebook exécuté avec succès !")
    except Exception as e:
        st.error(f"❌ Erreur lors de l'exécution du notebook : {e}")
        
        # Provide specific guidance for common errors
        error_str = str(e).lower()
        if "batch_shape" in error_str or "inputlayer" in error_str:
            st.error("💡 **Problème de compatibilité détecté :** Il semble y avoir un problème de version entre TensorFlow/Keras.")
            st.info("🔧 **Solutions possibles :**")
            st.info("1. Vérifiez que vous utilisez la même version de TensorFlow que lors de l'entraînement")
            st.info("2. Ou réentraînez les modèles avec votre version actuelle de TensorFlow")
            st.code("pip install tensorflow==2.15.0  # Exemple de version compatible")
        elif "filenotfounderror" in error_str:
            st.error("💡 **Fichiers manquants :** Certains modèles ou scalers sont introuvables.")
            st.info("🔧 Vérifiez que tous les modèles et scalers sont présents dans les dossiers models/ et scalers/")
        elif "keyerror" in error_str:
            st.error("💡 **Problème de données :** Certaines fonctionnalités n'ont pas pu être traitées.")
            st.info("🔧 Le notebook continuera avec les modèles disponibles.")
            st.info("📝 Vérifiez les messages dans le notebook pour plus de détails.")
        
        # Don't stop execution if it's just a KeyError - let the notebook handle it
        if "keyerror" not in error_str:
            st.stop()

    # Étape 3 : Charger le fichier CSV généré par le notebook
    try:
        # Charger les résultats
        results = pd.read_csv(results_path)
        st.success("✅ Résultats chargés avec succès !")
        
        # Ajouter un contrôle pour le nombre de jours à afficher
        total_days = len(results)
        st.subheader("⏱️ Configuration de l'affichage")
        selected_days = st.slider(
            "Nombre de jours à analyser", 
            min_value=1, 
            max_value=total_days, 
            value=total_days,
            help="Faites glisser pour ajuster le nombre de jours affichés dans les analyses"
        )
        
        # Filtrer les résultats selon le nombre de jours sélectionnés
        filtered_results = results.iloc[:selected_days]
        
        # Afficher le message sur le nombre de jours sélectionnés
        st.info(f"📅 Affichage des prédictions pour les {selected_days} premiers jours sur un total de {total_days} jours")

        # Calculer les KPI sur les résultats filtrés
        filtered_results["Generation > Consommation"] = filtered_results["generation_predite"] > filtered_results["consommation_predite"]
        jours_generation_sup = filtered_results["Generation > Consommation"].sum()
        jours_generation_inf = len(filtered_results) - jours_generation_sup

        pourcentage_generation_sup = (jours_generation_sup / len(filtered_results)) * 100
        pourcentage_generation_inf = 100 - pourcentage_generation_sup

        # Afficher les KPI
        st.subheader("📈 Indicateurs Clés de Performance (KPI)")
        col1, col2 = st.columns(2)
        col1.metric("Jours où la Génération > Consommation", f"{jours_generation_sup} jours", f"{pourcentage_generation_sup:.2f} %")
        col2.metric("Jours où la Génération < Consommation", f"{jours_generation_inf} jours", f"{pourcentage_generation_inf:.2f} %")        
        
        # Afficher les dates spécifiques où Génération > Consommation et Génération < Consommation
        st.subheader("📅 Dates Détaillées")
        
        # Séparer les données selon les conditions
        dates_generation_sup = filtered_results[filtered_results["Generation > Consommation"] == True]["date"].tolist()
        dates_generation_inf = filtered_results[filtered_results["Generation > Consommation"] == False]["date"].tolist()
        
        # Créer deux colonnes pour afficher les dates
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🟢 Dates où Génération > Consommation**")
            if dates_generation_sup:
                # Créer un DataFrame pour un meilleur affichage
                df_sup = pd.DataFrame({
                    'Date': dates_generation_sup,
                    'Statut': ['✅ Surplus'] * len(dates_generation_sup)
                })
                st.dataframe(df_sup, use_container_width=True, hide_index=True)
                
                # Option pour télécharger les dates
                csv_sup = df_sup.to_csv(index=False)
                st.download_button(
                    label="💾 Télécharger les dates (Génération > Consommation)",
                    data=csv_sup,
                    file_name="dates_generation_superieure.csv",
                    mime="text/csv"
                )
            else:
                st.info("Aucune date avec génération supérieure à la consommation")
        
        with col2:
            st.write("**🔴 Dates où Génération < Consommation**")
            if dates_generation_inf:
                # Créer un DataFrame pour un meilleur affichage
                df_inf = pd.DataFrame({
                    'Date': dates_generation_inf,
                    'Statut': ['❌ Déficit'] * len(dates_generation_inf)
                })
                st.dataframe(df_inf, use_container_width=True, hide_index=True)
                
                # Option pour télécharger les dates
                csv_inf = df_inf.to_csv(index=False)
                st.download_button(
                    label="💾 Télécharger les dates (Génération < Consommation)",
                    data=csv_inf,
                    file_name="dates_generation_inferieure.csv",
                    mime="text/csv"
                )
            else:
                st.info("Aucune date avec génération inférieure à la consommation")
        
        # Ajouter un graphique en barres pour visualiser la répartition par mois (si les données couvrent plusieurs mois)
        if len(filtered_results) > 30:  # Si plus de 30 jours, afficher la répartition mensuelle
            st.subheader("📊 Répartition Mensuelle des Surplus/Déficits")
            
            # Convertir les dates en datetime si ce n'est pas déjà fait
            filtered_results['date'] = pd.to_datetime(filtered_results['date'])
            filtered_results['month'] = filtered_results['date'].dt.strftime('%Y-%m')
            
            # Grouper par mois et calculer les statistiques
            monthly_stats = filtered_results.groupby('month').agg({
                'Generation > Consommation': 'sum',
                'date': 'count'
            }).rename(columns={'date': 'total_days', 'Generation > Consommation': 'surplus_days'})
            monthly_stats['deficit_days'] = monthly_stats['total_days'] - monthly_stats['surplus_days']
            monthly_stats['surplus_percentage'] = (monthly_stats['surplus_days'] / monthly_stats['total_days']) * 100;
            
            # Créer le graphique
            fig_monthly = go.Figure()
            
            fig_monthly.add_trace(go.Bar(
                x=monthly_stats.index,
                y=monthly_stats['surplus_days'],
                name='Jours avec Surplus',
                marker_color='#2E8B57',
                text=monthly_stats['surplus_days'],
                textposition='auto'
            ))
            
            fig_monthly.add_trace(go.Bar(
                x=monthly_stats.index,
                y=monthly_stats['deficit_days'],
                name='Jours avec Déficit',
                marker_color='#FF6B6B',
                text=monthly_stats['deficit_days'],
                textposition='auto'
            ))
            
            fig_monthly.update_layout(
                title='Répartition Mensuelle: Surplus vs Déficit Énergétique',
                xaxis_title='Mois',
                yaxis_title='Nombre de Jours',
                barmode='stack',
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig_monthly, use_container_width=True)
            
            # Afficher le tableau des statistiques mensuelles
            st.write("**📋 Statistiques Mensuelles Détaillées**")
            monthly_display = monthly_stats.copy()
            monthly_display['surplus_percentage'] = monthly_display['surplus_percentage'].round(2)
            monthly_display.columns = ['Jours Surplus', 'Total Jours', 'Jours Déficit', '% Surplus']
            st.dataframe(monthly_display, use_container_width=True)

        # Afficher les tableaux
        st.subheader("🔍 Détails des Prédictions")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Génération Prédite**")
            st.dataframe(filtered_results[["date", "generation_predite"]].set_index("date"))
        with col2:
            st.write("**Consommation Prédite**")
            st.dataframe(filtered_results[["date", "consommation_predite"]].set_index("date"))

        # Afficher les graphiques de prédictions
        display_prediction_plots(filtered_results)        # Initialize chatbot context
        chatbot = get_chatbot()
        chatbot.update_context(filtered_results)
        
        # Debug: Check if method exists
        if not hasattr(chatbot, 'get_predefined_prompts'):
            st.error("❌ Erreur: La méthode get_predefined_prompts n'existe pas. Veuillez redémarrer l'application.")
            st.info("🔄 Si le problème persiste, veuillez recharger la page complètement (Ctrl+F5).")
            st.stop()
        
        # Chatbot interaction with predefined prompts
        st.subheader("💬 Assistant IA pour l'Analyse Énergétique")
        
        # Get predefined prompts with error handling
        try:
            predefined_prompts = chatbot.get_predefined_prompts()
        except AttributeError as e:
            st.error(f"❌ Erreur AttributeError: {str(e)}")
            st.info("🔄 Veuillez redémarrer l'application Streamlit.")
            st.stop()
          # Create a more user-friendly display
        st.markdown("**Sélectionnez une question pour obtenir une analyse détaillée :**")
        
        # Create options for selectbox
        prompt_options = list(predefined_prompts.keys())
        selected_prompt_title = st.selectbox(
            "Choisissez votre question :",
            options=prompt_options,
            index=0,
            help="Sélectionnez la question qui vous intéresse le plus"
        )
        
        # Display description of selected prompt
        if selected_prompt_title:
            prompt_info = predefined_prompts[selected_prompt_title]
            st.info(f"ℹ️ {prompt_info['description']}")
        
        # Generate response button
        if st.button("🤖 Obtenir l'Analyse", use_container_width=True):
            if selected_prompt_title:
                selected_prompt = predefined_prompts[selected_prompt_title]["prompt"]
                
                with st.spinner("🔄 Analyse en cours..."):
                    response = chatbot.generate_response(selected_prompt)
                
                # Display response in an expandable container
                with st.expander(f"📋 Réponse : {selected_prompt_title}", expanded=True):
                    st.markdown(response)
                
                # Add option to ask follow-up questions
                st.markdown("---")
                st.markdown("**💡 Questions de suivi suggérées :**")
                
                # Suggest related prompts
                current_category = predefined_prompts[selected_prompt_title]['category']
                related_prompts = [title for title, prompt in predefined_prompts.items() 
                                 if prompt['category'] == current_category and title != selected_prompt_title]
                
                if related_prompts:
                    for related_prompt in related_prompts[:3]:  # Show max 3 related prompts
                        if st.button(f"🔗 {related_prompt}", key=f"related_{related_prompt}"):
                            related_response = chatbot.generate_response(predefined_prompts[related_prompt]["prompt"])
                            st.markdown(f"**{related_prompt} :**")
                            st.markdown(related_response)
            else:
                st.warning("⚠️ Veuillez sélectionner une question.")
    
    except FileNotFoundError:
        st.error("❌ Les résultats n'ont pas été générés ou le fichier est introuvable.")