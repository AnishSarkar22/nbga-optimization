# import streamlit as st
# from tsp import tsp
# from ligand_optimization import ligand

# st.sidebar.title("Navigation")
# page = st.sidebar.selectbox("Choose a page", ["NBGA comparison", "Ligand Optimization"])

# if page == "NBGA comparison":
#     tsp.main()
# elif page == "Ligand Optimization":
#     ligand.main()

# ------------------------------------------------------------

import streamlit as st

st.title("Welcome to the Optimization App")
st.markdown("""
### 🧰 Available Tools

- **TSP Algorithm Comparison**  
  Compare four advanced algorithms (NBGA, SWAP_GATSP, OX_SIM, MOC_SIM) on classic Traveling Salesman Problem datasets.  
  - Visualize and compare best tour distances and execution times.
  - Supports both coordinate-based and explicit distance matrix TSP files.
  - Interactive plots and performance tables for in-depth analysis.

- **NBGA Ligand Optimization**  
  Explore the Neighborhood-Based Genetic Algorithm (NBGA) for optimizing ligand-protein binding.  
  - Simulates ligand evolution to minimize binding energy using a genetic algorithm.
  - Visualizes the fitness (energy) evolution across generations.
  - Adjustable parameters for population size, generations, and ligand structure.

Select a tool from the sidebar to get started!
""")