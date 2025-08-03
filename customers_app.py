import streamlit as st
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx
import streamlit_bokeh
from bokeh.plotting import figure, from_networkx, show
import random

random_state = 1701

st.set_page_config(page_title="Customer Segmentation and Market Basket Analysis", layout="wide", page_icon="🛒")

st.header('Customer Segmentation and Market Basket Analysis')
css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:2rem;
    padding-right: 20px;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)

@st.cache_data
def load_data():
    purchases = pd.read_csv("Superstore.csv", encoding="utf-7")
    purchases = purchases.replace('\\+AC0-', '-', regex=True)
    purchases['Days Since Last Order'] = (max(pd.to_datetime(purchases['Order Date'], format='%d-%m-%Y')) - pd.to_datetime(purchases['Order Date'], format='%d-%m-%Y')).dt.days
    purchases['Profit'] = purchases['Profit'].astype('float')
    purchases['Customer Savings'] = (1-purchases['Discount'].astype('float'))*purchases['Sales']

    customers = purchases.groupby('Customer ID').agg({'Days Since Last Order': 'min', 'Order ID': ['count','nunique'], 'Ship Mode':','.join, 'Segment': 'first', 'Region': 'first', 'Product Name': ','.join, 'Category':','.join, 'Subcategory':','.join, 'Sales':'sum', 'Quantity':'sum', 'Customer Savings':'sum', 'Profit':'sum'}).reset_index()
    customers.columns = ['Customer ID','Days Since Last Order','Order ID', 'Number of Orders','Ship Mode', 'Segment', 'Region', 'Product Names', 'Categories', 'Subcategories', 'Total Spent', 'Number of Items', 'Customer Savings', 'Retailer Profit']
    customers['Savings per Order'] = customers['Customer Savings'] / customers['Number of Orders']
    customers['Items per Order'] = customers['Number of Items'] / customers['Number of Orders']
    customers['Spend per Order'] = customers['Total Spent'] / customers['Number of Orders']
    customers['Retailer Profit per Order'] = customers['Retailer Profit'] / customers['Number of Orders']
    # Number of different types of items ordered
    customers['Product Variety'] = customers['Product Names'].apply(lambda x: len(set(x.split(','))))
    customers['Category Variety'] = customers['Categories'].apply(lambda x: len(set(x.split(','))))
    customers['Subcategory Variety'] = customers['Subcategories'].apply(lambda x: len(set(x.split(','))))
    # Percent of items where customer chose first class, second class, or same-day shipping
    customers['Fast Shipping'] = (customers['Order ID'] - customers['Ship Mode'].apply(lambda x: x.count('Standard Class'))) / customers['Order ID']
    return purchases, customers

@st.cache_data
def process_customer_data(customers):
    customers_processed = customers.copy().set_index('Customer ID')
    # Log-scale Days Since Last Order and Total Spent since they are highly skewed. 
    # Also flip the sign of Days Since Last Order so that higher values indicate more recent purchases (in line with higher values being 'good' for other metrics)
    customers_processed['Days Since Last Order'] = -np.log(customers_processed['Days Since Last Order']+1)
    customers_processed['Total Spent'] = np.log(customers_processed['Total Spent']+1)
    customers_processed = customers_processed[['Number of Orders', 'Total Spent', 'Days Since Last Order', 'Items per Order', 'Savings per Order', 'Product Variety', 'Subcategory Variety']]
    scaler = StandardScaler()
    customers_scaled = pd.DataFrame(scaler.fit_transform(customers_processed), index=customers_processed.index, columns=customers_processed.columns)
    return customers_scaled

@st.cache_resource
def cluster_model(customers_scaled):
    agg_cluster = AgglomerativeClustering(n_clusters=None, metric='euclidean', linkage='ward', compute_full_tree=True, distance_threshold=25)
    agg_cluster.fit_predict(customers_scaled)
    return agg_cluster
    
@st.cache_data
def product_associations(purchases):
    orders = purchases.merge(customers[['Customer ID', 'Cluster']], how='left').groupby("Order ID").agg(
    {'Days Since Last Order': 'first', 'Ship Mode':'first', 'Region': 'first', 'Product Name': list, 'Category':list, 'Subcategory':list, 
     'Sales':'sum', 'Quantity':'sum', 'Customer Savings':'sum', 'Profit':'sum', 'Customer ID':'first', 'Cluster':'first'}
    ).reset_index()
    
    te = TransactionEncoder()
    subcat_encoded = pd.DataFrame(te.fit_transform(orders['Subcategory']), columns=te.columns_)
    subcat_apriori = apriori(subcat_encoded, min_support=0.01, use_colnames=True)
    rules_apriori = association_rules(subcat_apriori, metric="confidence", min_threshold=0)
    rules_apriori['antecedents'] = rules_apriori['antecedents'].apply(lambda x:', '.join(sorted(list(x))))
    rules_apriori['consequents'] = rules_apriori['consequents'].apply(lambda x:', '.join(sorted(list(x))))
    return rules_apriori.sort_values('support', ascending=False)
    
def get_product_subcategory(product, purchases):
    return list(purchases.loc[purchases['Product Name']==product,'Subcategory'])[0]
    
purchases, customers = load_data()
# The "Staples" entries are categorized inconsistently. They should all have the same category and subcategory. I chose "Office Supplies > Fasteners"
purchases.loc[purchases['Product Name']=="Staples",['Category', 'Subcategory']] = ['Office Supplies', 'Fasteners']

customers_scaled = process_customer_data(customers)
agg_cluster = cluster_model(customers_scaled)
cluster_labels = {0: "Bulk buyers", 1: "Disengaged", 2: "Loyal", 3: "Moderately engaged"}
customers['Cluster'] = [cluster_labels[c] for c in agg_cluster.labels_]
cluster_colors = {"Loyal":"green", "Bulk buyers":"blue", "Moderately engaged":"orange", "Disengaged":"red"}

rules = product_associations(purchases)

tab1, tab2, tab3 = st.tabs(["Interactive Segment Comparison", "Metrics by Segment", "Market Basket Analysis"])
with tab1:
    col1, col2, col3 = st.columns([1, 3, 2])
    with col1:
        st.markdown("### Compare metrics by segment")
        metric1 = st.selectbox("Metric 1:", customers_scaled.columns, index=0)
        metric2 = st.selectbox("Metric 2:", customers_scaled.columns, index=1)
    with col2:
        if metric1 in ['Number of Orders', 'Product Variety', 'Subcategory Variety']:
            fig = px.strip(customers, x=metric1, y=metric2, color='Cluster',
            category_orders = {'Cluster': ["Loyal", "Bulk buyers", "Moderately engaged", "Disengaged"]},
            color_discrete_map = cluster_colors,
            )
        else:
            fig = px.scatter(customers, x=metric1, y=metric2, color='Cluster',
                category_orders = {'Cluster': ["Loyal", "Bulk buyers", "Moderately engaged", "Disengaged"]},
                color_discrete_map = cluster_colors,
            )
        fig.update_layout(height=400)
        fig.update_traces(opacity=.8)
        st.plotly_chart(fig)
    with col3:
        st.markdown("### Segment Averages:")
        st.dataframe(customers.groupby('Cluster').agg({metric1: lambda x: np.mean(x).round(2), metric2: lambda x: np.mean(x).round(2)}).reindex(["Loyal", "Bulk buyers", "Moderately engaged", "Disengaged"]))
    st.markdown("---")
    st.markdown("### Customer Lookup")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:    
        cust_id = st.selectbox("Customer ID:", customers["Customer ID"].sort_values())
    with col2:
        clust = customers[customers["Customer ID"] == cust_id]["Cluster"].to_string(index=False)
        st.markdown("### Segment: :" + cluster_colors[clust]+ "[" + clust + "]")
    with col3:
        st.metric("Number of orders", customers[customers["Customer ID"] == cust_id]["Number of Orders"])
    with col4:
        st.metric("Items purchased", customers[customers["Customer ID"] == cust_id]["Number of Items"])
    with col5:
        st.metric("Total spent", f'${customers.loc[customers["Customer ID"] == cust_id,"Total Spent"].item():.2f}')

with tab2:
    col1, col2, col3, col4 = st.columns(4, border = True)
    with col1:
        st.markdown("### Loyal Segment:")
        seg = customers[customers['Cluster']=="Loyal"]
        colA, colB, colC = st.columns(3)
        with colA:
            st.metric("Customers", len(seg))
        with colB:
            st.metric("Profit per order:", f"${(seg['Retailer Profit per Order'].sum()/len(seg)):.2f}")
        with colC:
            st.metric("Profit per customer:", f"${(seg['Retailer Profit per Order']*seg['Number of Orders']).sum()/len(seg):.2f}")
        for m in customers_scaled.columns:
            fig = px.histogram(seg, x=m,color='Cluster', color_discrete_map=cluster_colors, range_x=[min(customers[m]), max(customers[m])])
            fig = fig.update_layout(yaxis_title="", height=300, title=m, legend_visible=False)
            st.plotly_chart(fig)
    with col2:
        st.markdown("### Bulk Buyers Segment:")
        seg = customers[customers['Cluster']=="Bulk buyers"]
        colA, colB, colC = st.columns(3)
        with colA:
            st.metric("Customers", len(seg))
        with colB:
            st.metric("Profit per order:", f"${(seg['Retailer Profit per Order'].sum()/len(seg)):.2f}")
        with colC:
            st.metric("Profit per customer:", f"${(seg['Retailer Profit per Order']*seg['Number of Orders']).sum()/len(seg):.2f}")
        for m in customers_scaled.columns:
            fig = px.histogram(seg, x=m,color='Cluster', color_discrete_map=cluster_colors, range_x=[min(customers[m]), max(customers[m])])
            fig = fig.update_layout(yaxis_title="", height=300, title=m, legend_visible=False)
            st.plotly_chart(fig)
    with col3:
        st.markdown("### Moderately Engaged Segment:") 
        seg = customers[customers['Cluster']=="Moderately engaged"]
        colA, colB, colC = st.columns(3)
        with colA:
            st.metric("Customers", len(seg))
        with colB:
            st.metric("Profit per order:", f"${(seg['Retailer Profit per Order'].sum()/len(seg)):.2f}")
        with colC:
            st.metric("Profit per customer:", f"${(seg['Retailer Profit per Order']*seg['Number of Orders']).sum()/len(seg):.2f}")
        for m in customers_scaled.columns:
            fig = px.histogram(seg, x=m,color='Cluster', color_discrete_map=cluster_colors, range_x=[min(customers[m]), max(customers[m])])
            fig = fig.update_layout(yaxis_title="", height=300, title=m, legend_visible=False)
            st.plotly_chart(fig)
    with col4:
        st.markdown("### Disengaged Segment:")
        seg = customers[customers['Cluster']=="Disengaged"]
        colA, colB, colC = st.columns(3)
        with colA:
            st.metric("Customers", len(seg))
        with colB:
            st.metric("Profit per order:", f"${(seg['Retailer Profit per Order'].sum()/len(seg)):.2f}")
        with colC:
            st.metric("Profit per customer:", f"${(seg['Retailer Profit per Order']*seg['Number of Orders']).sum()/len(seg):.2f}")
        for m in customers_scaled.columns:
            fig = px.histogram(seg, x=m,color='Cluster', color_discrete_map=cluster_colors, range_x=[min(customers[m]), max(customers[m])])
            fig = fig.update_layout(yaxis_title="", height=300, title=m, legend_visible=False)
            st.plotly_chart(fig)
with tab3:
    st.markdown("## Association Rule Explorer")
    col1, col2, col3 = st.columns([1,4, 2])
    with col1:
        min_support = st.slider("Minimum support", min_value=0.0, max_value=0.05, value=0.01, help="Frequency of items being purchased together")
        min_confidence = st.slider("Minimum confidence", min_value=0.0, max_value=0.3, value=0.25, help="Probability that someone who buys the first item ('antecedent') will also buy the second ('consequent')")
        min_lift = st.slider("Minimum lift", min_value=0.0, max_value=1.5, value=1.0, help="Measure of how likely it is that purchase decisions for these products are not independent")
    with col2:
        filtered_rules = rules[
            ((rules['support'] >= min_support) &
            (rules['confidence'] >= min_confidence) &
            (rules['lift'] >= min_lift))
        ]
        if len(filtered_rules) == 0:
            st.markdown("## No rules found for given criteria")
        else:
            st.dataframe(filtered_rules[['antecedents','consequents', 'support', 'confidence', 'lift', 'leverage', 'conviction']], hide_index=True)
    category_dict = {'Accessories':'Technology', 'Appliances':'Office Supplies', 'Art':'Office Supplies', 'Binders':'Office Supplies', 'Bookcases':'Furniture', 'Chairs':'Furniture', 'Copiers':'Technology', 'Envelopes':'Office Supplies', 'Fasteners':'Office Supplies', 'Furnishings':'Furniture', 'Labels':'Office Supplies', 'Machines':'Technology', 'Paper':'Office Supplies', 'Phones':'Technology', 'Storage':'Office Supplies', 'Supplies':'Office Supplies', 'Tables':'Furniture'}
    category_colors = {'Furniture':'orange', 'Office Supplies':'blue', 'Technology': 'green'}
    with col3:
        st.markdown("**Rule Network Visualization:**\nHover over a node to see what it represents!")
        G = nx.Graph()        
        # Add nodes and edges
        for i, rule in filtered_rules.iterrows():
            if ',' not in rule['antecedents']:
                if rule['antecedents'] not in G.nodes:
                    G.add_node(rule['antecedents'], category=category_dict[rule['antecedents']])
        for i, rule in filtered_rules.iterrows():  
            if ',' not in rule['antecedents'] and ',' not in rule['consequents']:
                if rule['consequents'] not in G.nodes:
                    G.add_node(rule['consequents'], category=category_dict[rule['consequents']])
                G.add_edge(rule['antecedents'], rule['consequents'], weight=rule['support'])
        p = figure(x_range=(-2, 2), y_range=(-2,2),
                   x_axis_location=None, y_axis_location=None,
                   tools="hover", tooltips="@index\n(@category)")
        p.grid.grid_line_color = None
        from bokeh.models import Circle, HoverTool
        graph = from_networkx(G, nx.spring_layout, scale=1, center=(0,0))
        graph.node_renderer.glyph = Circle(radius=0.08)
        graph.node_renderer.data_source.data['colors'] = [category_colors[category_dict[n]] for n in G.nodes]
        graph.node_renderer.glyph.update(fill_color="colors")
        p.renderers.append(graph)
        streamlit_bokeh.streamlit_bokeh(p, use_container_width=False)
    st.markdown("---")
    col1, col2 = st.columns([2, 4])
    with col1:
        cart = st.multiselect("Products", options=sorted(purchases["Product Name"].unique()))
    with col2:
        all_products = purchases[['Product Name', 'Subcategory']].drop_duplicates()
        categories = ', '.join(list(dict.fromkeys(sorted([get_product_subcategory(p, all_products) for p in cart]))))
        sampled_items = []
        if categories in list(filtered_rules['antecedents']):
            possible_categories = []
            for i, rule in filtered_rules[filtered_rules['antecedents']==categories].iterrows():
                possible_categories += rule['consequents'].split(', ')
            possible_items = all_products[all_products['Subcategory'].isin(possible_categories)]
            sampled_items = random.sample(list(possible_items['Product Name']), 3)
            st.markdown("You may also like:")
            for s in sampled_items:               
                st.badge(s)
        else:
            "No recommended product types found based on current basket"