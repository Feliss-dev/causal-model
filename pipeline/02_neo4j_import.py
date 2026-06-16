"""
02_neo4j_import.py
==================
Import HIN vào Neo4j và chạy GDS graph algorithms.

Thay đổi so với phiên bản cũ:
- Đọc credentials từ .env (không hard-code password)
- Xoá Comment node, WROTE, HAS_COMMENT, CROSS_POST khỏi import và GDS
- Dùng feature column thực (post_count, fake_ratio_real, avg_score, ...)
- Cập nhật local NetworkX fallback tương ứng
"""

import os
import sys
import pandas as pd
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv không bắt buộc; dùng biến môi trường hoặc default

from neo4j import GraphDatabase

# ===================== CONFIG =====================
INPUT_DIR  = os.path.join("data", "processed")
OUTPUT_DIR = os.path.join("data", "processed")

NEO4J_URI      = os.environ.get("NEO4J_URI",      "bolt://localhost:7887")
NEO4J_USER     = os.environ.get("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password123")


# ===================== CONNECTION =====================

def check_neo4j_connection():
    print(f"Kết nối Neo4j tại {NEO4J_URI}...")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("Kết nối Neo4j thành công!")
        return driver
    except Exception as e:
        print(f"\n[CẢNH BÁO] Neo4j không kết nối được: {e}")
        print("Hãy đảm bảo Docker Desktop đang chạy và thực hiện: docker compose up -d")
        print("Chuyển sang local graph analysis bằng NetworkX...")
        return None


# ===================== NEO4J IMPORT =====================

def import_to_neo4j(driver):
    print("\nBắt đầu import dữ liệu vào Neo4j...")

    # Đọc CSV nodes
    posts_df      = pd.read_csv(os.path.join(INPUT_DIR, "posts.csv"))
    users_df      = pd.read_csv(os.path.join(INPUT_DIR, "users.csv"))
    subreddits_df = pd.read_csv(os.path.join(INPUT_DIR, "subreddits.csv"))
    domains_df    = pd.read_csv(os.path.join(INPUT_DIR, "domains.csv"))
    images_df     = pd.read_csv(os.path.join(INPUT_DIR, "images.csv"))

    # Đọc CSV edges
    posted_by = pd.read_csv(os.path.join(INPUT_DIR, "posted_by.csv"))
    posted_in = pd.read_csv(os.path.join(INPUT_DIR, "posted_in.csv"))
    links_to  = pd.read_csv(os.path.join(INPUT_DIR, "links_to.csv"))
    has_image = pd.read_csv(os.path.join(INPUT_DIR, "has_image.csv"))
    member_of = pd.read_csv(os.path.join(INPUT_DIR, "member_of.csv"))

    with driver.session() as session:

        # Xoá dữ liệu cũ
        print("Xoá dữ liệu cũ trong Neo4j...")
        session.run("MATCH (n) DETACH DELETE n")

        # Tạo constraints
        print("Tạo constraints...")
        constraints = [
            "CREATE CONSTRAINT post_id_uq IF NOT EXISTS FOR (p:Post) REQUIRE p.post_id IS UNIQUE",
            "CREATE CONSTRAINT user_id_uq IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE",
            "CREATE CONSTRAINT sub_id_uq  IF NOT EXISTS FOR (s:Subreddit) REQUIRE s.sub_id IS UNIQUE",
            "CREATE CONSTRAINT dom_id_uq  IF NOT EXISTS FOR (d:Domain) REQUIRE d.domain_id IS UNIQUE",
            "CREATE CONSTRAINT img_id_uq  IF NOT EXISTS FOR (i:Image) REQUIRE i.img_id IS UNIQUE",
        ]
        for c in constraints:
            session.run(c)

        def run_batch(tx, query, rows, batch_size=500):
            for i in range(0, len(rows), batch_size):
                tx.run(query, rows=rows[i: i + batch_size])

        # ---- Post ----
        print("Import Post nodes...")
        post_rows = [
            {
                "post_id":      str(r["post_id"]),
                "title":        str(r["title"]),
                "score":        int(r["score"]),
                "upvote_ratio": float(r["upvote_ratio"]),
                "num_comments": int(r["num_comments"]),
                "has_image":    bool(r["has_image"]),
                "label_2way":   int(r["label_2way"]),
                "label_6way":   int(r["label_6way"]),
                "split":        str(r["split"]),
            }
            for _, r in posts_df.iterrows()
        ]
        session.execute_write(run_batch,
            "UNWIND $rows AS r MERGE (p:Post {post_id: r.post_id}) SET p += r",
            post_rows)

        # ---- User ----
        print("Import User nodes...")
        user_rows = [
            {
                "user_id":          str(r["user_id"]),
                "name":             str(r["name"]),
                "post_count":       int(r["post_count"]),
                "avg_score":        float(r["avg_score"]),
                "avg_upvote_ratio": float(r["avg_upvote_ratio"]),
                "fake_rate":        float(r["fake_rate"]),
            }
            for _, r in users_df.iterrows()
        ]
        session.execute_write(run_batch,
            "UNWIND $rows AS r MERGE (u:User {user_id: r.user_id}) SET u += r",
            user_rows)

        # ---- Subreddit ----
        print("Import Subreddit nodes...")
        sub_rows = [
            {
                "sub_id":          str(r["sub_id"]),
                "name":            str(r["name"]),
                "post_count":      int(r["post_count"]),
                "fake_ratio_real": float(r["fake_ratio_real"]),
                "avg_score":       float(r["avg_score"]),
            }
            for _, r in subreddits_df.iterrows()
        ]
        session.execute_write(run_batch,
            "UNWIND $rows AS r MERGE (s:Subreddit {sub_id: r.sub_id}) SET s += r",
            sub_rows)

        # ---- Domain ----
        print("Import Domain nodes...")
        domain_rows = [
            {
                "domain_id":       str(r["domain_id"]),
                "url_domain":      str(r["url_domain"]),
                "post_count":      int(r["post_count"]),
                "fake_ratio_real": float(r["fake_ratio_real"]),
                "avg_upvote_ratio": float(r["avg_upvote_ratio"]),
            }
            for _, r in domains_df.iterrows()
        ]
        session.execute_write(run_batch,
            "UNWIND $rows AS r MERGE (d:Domain {domain_id: r.domain_id}) SET d += r",
            domain_rows)

        # ---- Image ----
        print("Import Image nodes...")
        img_rows = [
            {
                "img_id":    str(r["img_id"]),
                "post_id":   str(r["post_id"]),
                "has_image": bool(r["has_image"]),
                "image_url": str(r["image_url"]),
            }
            for _, r in images_df.iterrows()
        ]
        session.execute_write(run_batch,
            "UNWIND $rows AS r MERGE (i:Image {img_id: r.img_id}) SET i += r",
            img_rows)

        # ---- Edges ----
        print("Import POSTED_BY relationships...")
        edge_rows = [{"post_id": str(r["post_id"]), "user_id": str(r["user_id"])}
                     for _, r in posted_by.iterrows()]
        session.execute_write(run_batch, """
            UNWIND $rows AS r
            MATCH (p:Post {post_id: r.post_id})
            MATCH (u:User {user_id: r.user_id})
            MERGE (p)-[:POSTED_BY]->(u)
        """, edge_rows)

        print("Import POSTED_IN relationships...")
        edge_rows = [{"post_id": str(r["post_id"]), "sub_id": str(r["sub_id"])}
                     for _, r in posted_in.iterrows()]
        session.execute_write(run_batch, """
            UNWIND $rows AS r
            MATCH (p:Post {post_id: r.post_id})
            MATCH (s:Subreddit {sub_id: r.sub_id})
            MERGE (p)-[:POSTED_IN]->(s)
        """, edge_rows)

        print("Import LINKS_TO relationships...")
        edge_rows = [{"post_id": str(r["post_id"]), "domain_id": str(r["domain_id"])}
                     for _, r in links_to.iterrows()]
        session.execute_write(run_batch, """
            UNWIND $rows AS r
            MATCH (p:Post {post_id: r.post_id})
            MATCH (d:Domain {domain_id: r.domain_id})
            MERGE (p)-[:LINKS_TO]->(d)
        """, edge_rows)

        print("Import HAS_IMAGE relationships...")
        edge_rows = [{"post_id": str(r["post_id"]), "img_id": str(r["img_id"]),
                      "is_primary": bool(r["is_primary"])}
                     for _, r in has_image.iterrows()]
        session.execute_write(run_batch, """
            UNWIND $rows AS r
            MATCH (p:Post {post_id: r.post_id})
            MATCH (i:Image {img_id: r.img_id})
            MERGE (p)-[:HAS_IMAGE {is_primary: r.is_primary}]->(i)
        """, edge_rows)

        print("Import MEMBER_OF relationships...")
        edge_rows = [{"user_id": str(r["user_id"]), "sub_id": str(r["sub_id"]),
                      "activity_level": int(r["activity_level"])}
                     for _, r in member_of.iterrows()]
        session.execute_write(run_batch, """
            UNWIND $rows AS r
            MATCH (u:User {user_id: r.user_id})
            MATCH (s:Subreddit {sub_id: r.sub_id})
            MERGE (u)-[:MEMBER_OF {activity_level: r.activity_level}]->(s)
        """, edge_rows)

        print("Import hoàn tất!")

        # ---- GDS Algorithms ----
        posts_df, users_df, domains_df = _run_gds(
            session, posts_df, users_df, domains_df, subreddits_df, images_df
        )

        # ---- Causal DAG Metadata (G-06) ----
        _store_causal_dag(session)

    # Lưu enriched CSVs (sau khi đã merge GDS results)
    print("\nLưu enriched CSV...")
    posts_df.to_csv(os.path.join(OUTPUT_DIR, "posts_enriched.csv"), index=False)
    domains_df.to_csv(os.path.join(OUTPUT_DIR, "domains_enriched.csv"), index=False)
    users_df.to_csv(os.path.join(OUTPUT_DIR, "users_enriched.csv"), index=False)
    subreddits_df.to_csv(os.path.join(OUTPUT_DIR, "subreddits_enriched.csv"), index=False)
    images_df.to_csv(os.path.join(OUTPUT_DIR, "images_enriched.csv"), index=False)
    print("Lưu enriched CSV xong!")


def _run_gds(session, posts_df, users_df, domains_df, subreddits_df, images_df):
    """Chạy GDS algorithms: PageRank, Louvain, Betweenness, FastRP."""
    print("\nChạy Neo4j GDS algorithms...")
    try:
        # Xoá graph projection cũ nếu tồn tại
        try:
            session.run("CALL gds.graph.drop('fakedditGraph', false) YIELD graphName")
        except Exception:
            pass

        # Project graph (chỉ 5 node type, 5 edge type — không có Comment/CROSS_POST)
        print("Project graph cho GDS...")
        session.run("""
            CALL gds.graph.project(
              'fakedditGraph',
              ['Post', 'User', 'Subreddit', 'Domain', 'Image'],
              {
                POSTED_BY: {type: 'POSTED_BY', orientation: 'UNDIRECTED'},
                POSTED_IN: {type: 'POSTED_IN', orientation: 'UNDIRECTED'},
                LINKS_TO:  {type: 'LINKS_TO',  orientation: 'UNDIRECTED'},
                HAS_IMAGE: {type: 'HAS_IMAGE',  orientation: 'UNDIRECTED'},
                MEMBER_OF: {type: 'MEMBER_OF',  orientation: 'UNDIRECTED'}
              }
            )
        """)

        # PageRank (uy tín Domain)
        print("Chạy PageRank...")
        session.run("""
            CALL gds.pageRank.write('fakedditGraph', {writeProperty: 'pagerank'})
        """)

        # Louvain (community detection trên Post)
        print("Chạy Louvain community detection...")
        session.run("""
            CALL gds.louvain.write('fakedditGraph', {writeProperty: 'community_id'})
        """)

        # Betweenness Centrality (User influence)
        print("Chạy Betweenness Centrality...")
        session.run("""
            CALL gds.betweenness.write('fakedditGraph', {writeProperty: 'betweenness'})
        """)

        # FastRP Graph Embeddings (64-dim) cho Post
        print("Chạy FastRP embeddings...")
        session.run("""
            CALL gds.fastRP.write('fakedditGraph', {
              embeddingDimension: 64,
              writeProperty: 'graph_embedding'
            })
        """)

        # Node Similarity (User-User SIMILAR_TO based on shared Subreddits)
        print("Chạy Node Similarity (User-User)...")
        try:
            # Drop old SIMILAR_TO relationships
            session.run("MATCH ()-[r:SIMILAR_TO]->() DELETE r")
            # Project a User-Subreddit bipartite graph for node similarity
            try:
                session.run("CALL gds.graph.drop('userSubGraph', false) YIELD graphName")
            except Exception:
                pass
            session.run("""
                CALL gds.graph.project(
                  'userSubGraph',
                  ['User', 'Subreddit'],
                  { MEMBER_OF: {type: 'MEMBER_OF', orientation: 'UNDIRECTED'} }
                )
            """)
            session.run("""
                CALL gds.nodeSimilarity.write('userSubGraph', {
                  writeRelationshipType: 'SIMILAR_TO',
                  writeProperty: 'similarity',
                  similarityCutoff: 0.1,
                  topK: 5
                })
            """)
            try:
                session.run("CALL gds.graph.drop('userSubGraph', false) YIELD graphName")
            except Exception:
                pass
            print("Node Similarity hoàn tất - SIMILAR_TO relationships đã lưu!")
        except Exception as e:
            print(f"  Node Similarity thất bại ({e}), bỏ qua...")

        print("GDS algorithms hoàn tất!")

        # Pull back GDS results
        print("Đọc GDS results từ Neo4j...")

        # Domain PageRank
        res = session.run(
            "MATCH (d:Domain) RETURN d.domain_id AS domain_id, d.pagerank AS pagerank"
        ).data()
        pr_df = pd.DataFrame(res)
        domains_df = domains_df.merge(pr_df, on="domain_id", how="left")
        domains_df["pagerank"] = domains_df["pagerank"].fillna(
            1.0 / max(len(domains_df), 1)
        )

        # Post community_id
        res = session.run(
            "MATCH (p:Post) RETURN p.post_id AS post_id, p.community_id AS community_id"
        ).data()
        comm_df = pd.DataFrame(res)
        posts_df = posts_df.merge(comm_df, on="post_id", how="left")
        posts_df["community_id"] = posts_df["community_id"].fillna(0).astype(int)

        # User betweenness
        res = session.run(
            "MATCH (u:User) RETURN u.user_id AS user_id, u.betweenness AS betweenness"
        ).data()
        btw_df = pd.DataFrame(res)
        users_df = users_df.merge(btw_df, on="user_id", how="left")
        users_df["betweenness"] = users_df["betweenness"].fillna(0.0)

        # FastRP for Posts
        print("Trích xuất FastRP embeddings...")
        res = session.run(
            "MATCH (p:Post) RETURN p.post_id AS post_id, p.graph_embedding AS graph_embedding"
        ).data()
        fastrp_dict = {str(item["post_id"]): item["graph_embedding"] for item in res}

        fastrp_list = []
        for pid in posts_df["post_id"]:
            emb = fastrp_dict.get(str(pid))
            if emb is None or not isinstance(emb, list) or len(emb) != 64:
                emb = list(np.zeros(64, dtype=np.float32))
            fastrp_list.append(emb)
        fastrp_arr = np.array(fastrp_list, dtype=np.float32)
        np.save(os.path.join(OUTPUT_DIR, "post_fastrp.npy"), fastrp_arr)
        print(f"Đã lưu post_fastrp.npy: shape={fastrp_arr.shape}")

        return posts_df, users_df, domains_df

    except Exception as e:
        print(f"\n[CẢNH BÁO] GDS thất bại (plugin chưa cài?): {e}")
        print("Chuyển sang local graph fallback...")
        run_local_graph_fallback()
        # Re-read enriched CSVs that fallback saved
        posts_df   = pd.read_csv(os.path.join(OUTPUT_DIR, "posts_enriched.csv"))
        users_df   = pd.read_csv(os.path.join(OUTPUT_DIR, "users_enriched.csv"))
        domains_df = pd.read_csv(os.path.join(OUTPUT_DIR, "domains_enriched.csv"))
        return posts_df, users_df, domains_df


def _store_causal_dag(session):
    """Lưu Causal DAG metadata vào Neo4j dưới dạng node :CausalDAG (G-06)."""
    print("\nLưu Causal DAG metadata vào Neo4j...")
    dag_definition = {
        "name": "FakedditCausalDAG",
        "version": "1.0",
        "description": (
            "Formal Causal DAG for Fakeddit HIN. "
            "Causal variables: Content(X), Image(I), Domain(D), Label(Y). "
            "Confounders: Subreddit(C1) -> X, Y; User(C2) -> X, Y. "
            "Causal paths: X->Y, I->Y, D->Y. "
            "Spurious paths (backdoor): C1->X, C1->Y; C2->X, C2->Y. "
            "Intervention targets: do(I=0), do(D=credible), do(C1=neutral)."
        ),
        "causal_variables": "Content, Image, Domain, Label",
        "confounders": "Subreddit(C1), User(C2)",
        "causal_paths": "Content->Label, Image->Label, Domain->Label",
        "spurious_paths": "Subreddit->Content, Subreddit->Label, User->Content, User->Label",
        "interventions": "do(Image=None), do(Domain=credible), do(Subreddit=neutral)",
        "ood_subreddits": "neutralnews, theonion",
        "grl_alpha": 2.0,
        "backdoor_adjustment": (
            "structural intervention: remove all edges incident on Subreddit (C1) "
            "in the causal branch + adversarial GRL on subreddit-ID. "
            "NOTE: C2 (User) and Domain label-history features are NOT cut — "
            "see GNN_NEUTRAL_DOMAIN/GNN_NEUTRAL_USER ablation in 06_train_gnn.py."
        ),
    }
    cypher = """
        MERGE (dag:CausalDAG {name: $name})
        SET dag += $props
        RETURN dag.name AS name
    """
    props = {k: str(v) for k, v in dag_definition.items() if k != "name"}
    result = session.run(cypher, name=dag_definition["name"], props=props).single()
    if result:
        print(f"Causal DAG node lưu thành công: {result['name']}")
    else:
        print("Causal DAG node đã tồn tại hoặc lỗi.")


# ===================== LOCAL FALLBACK (NetworkX) =====================

def run_local_graph_fallback():
    print("\n--- Local Graph Fallback (NetworkX) ---")
    import networkx as nx

    posts_df      = pd.read_csv(os.path.join(INPUT_DIR, "posts.csv"))
    users_df      = pd.read_csv(os.path.join(INPUT_DIR, "users.csv"))
    subreddits_df = pd.read_csv(os.path.join(INPUT_DIR, "subreddits.csv"))
    domains_df    = pd.read_csv(os.path.join(INPUT_DIR, "domains.csv"))
    images_df     = pd.read_csv(os.path.join(INPUT_DIR, "images.csv"))

    posted_by = pd.read_csv(os.path.join(INPUT_DIR, "posted_by.csv"))
    posted_in = pd.read_csv(os.path.join(INPUT_DIR, "posted_in.csv"))
    links_to  = pd.read_csv(os.path.join(INPUT_DIR, "links_to.csv"))
    has_image = pd.read_csv(os.path.join(INPUT_DIR, "has_image.csv"))
    member_of = pd.read_csv(os.path.join(INPUT_DIR, "member_of.csv"))

    G = nx.Graph()

    # Thêm nodes
    for _, r in posts_df.iterrows():
        G.add_node(str(r["post_id"]), ntype="Post")
    for _, r in users_df.iterrows():
        G.add_node(str(r["user_id"]), ntype="User")
    for _, r in subreddits_df.iterrows():
        G.add_node(str(r["sub_id"]), ntype="Subreddit")
    for _, r in domains_df.iterrows():
        G.add_node(str(r["domain_id"]), ntype="Domain")
    for _, r in images_df.iterrows():
        G.add_node(str(r["img_id"]), ntype="Image")

    # Thêm edges
    for _, r in posted_by.iterrows():
        G.add_edge(str(r["post_id"]), str(r["user_id"]), rel="POSTED_BY")
    for _, r in posted_in.iterrows():
        G.add_edge(str(r["post_id"]), str(r["sub_id"]), rel="POSTED_IN")
    for _, r in links_to.iterrows():
        G.add_edge(str(r["post_id"]), str(r["domain_id"]), rel="LINKS_TO")
    for _, r in has_image.iterrows():
        G.add_edge(str(r["post_id"]), str(r["img_id"]), rel="HAS_IMAGE")
    for _, r in member_of.iterrows():
        G.add_edge(str(r["user_id"]), str(r["sub_id"]), rel="MEMBER_OF")

    print(f"NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # 1. PageRank → Domain
    print("Tính PageRank...")
    pr = nx.pagerank(G, max_iter=100)
    domains_df["pagerank"] = domains_df["domain_id"].apply(
        lambda x: pr.get(str(x), 0.0)
    )
    mn, mx = domains_df["pagerank"].min(), domains_df["pagerank"].max()
    if mx > mn:
        domains_df["pagerank"] = 0.05 + 0.95 * (domains_df["pagerank"] - mn) / (mx - mn)
    else:
        domains_df["pagerank"] = 0.5

    # 2. Louvain → Post community_id
    print("Tính Louvain communities...")
    try:
        from networkx.algorithms.community import louvain_communities
        comms = list(louvain_communities(G, seed=42))
        comm_map = {node: idx for idx, comm in enumerate(comms) for node in comm}
        posts_df["community_id"] = posts_df["post_id"].apply(
            lambda x: comm_map.get(str(x), 0)
        )
    except Exception as e:
        print(f"  Louvain thất bại ({e}), gán community ngẫu nhiên...")
        posts_df["community_id"] = np.random.randint(0, 5, size=len(posts_df))

    # 3. Betweenness → User
    print("Tính Betweenness Centrality (k-sample)...")
    try:
        k = min(200, G.number_of_nodes())
        btw = nx.betweenness_centrality(G, k=k, seed=42)
        users_df["betweenness"] = users_df["user_id"].apply(
            lambda x: btw.get(str(x), 0.0)
        )
    except Exception as e:
        print(f"  Betweenness thất bại ({e}), gán 0...")
        users_df["betweenness"] = 0.0

    # 4. FastRP-style graph embedding (Post-centric, sparse)
    print("Tính FastRP local (sparse multi-hop)...")
    _compute_local_fastrp(G, posts_df)

    # 5. Node Similarity (User cosine sim based on shared subreddits via Jaccard)
    print("Tính Node Similarity local (User-User Jaccard)...")
    try:
        user_subs = {}
        for _, r in member_of.iterrows():
            uid = str(r["user_id"])
            sid = str(r["sub_id"])
            user_subs.setdefault(uid, set()).add(sid)
        # Write SIMILAR_TO as a CSV for reference (not Neo4j here)
        sim_rows = []
        user_ids_list = list(user_subs.keys())
        for i, u1 in enumerate(user_ids_list[:200]):  # limit pairs
            s1 = user_subs[u1]
            for u2 in user_ids_list[i+1:i+6]:
                s2 = user_subs[u2]
                if s1 and s2:
                    jaccard = len(s1 & s2) / len(s1 | s2)
                    if jaccard > 0.1:
                        sim_rows.append({"user_id_1": u1, "user_id_2": u2, "similarity": round(jaccard, 4)})
        pd.DataFrame(sim_rows).to_csv(os.path.join(OUTPUT_DIR, "user_similarity.csv"), index=False)
        print(f"Node Similarity: {len(sim_rows)} SIMILAR_TO pairs saved to user_similarity.csv")
    except Exception as e:
        print(f"  Node Similarity local thất bại ({e})")

    # Lưu enriched CSVs
    print("Lưu local enriched CSVs...")
    posts_df.to_csv(os.path.join(OUTPUT_DIR, "posts_enriched.csv"), index=False)
    domains_df.to_csv(os.path.join(OUTPUT_DIR, "domains_enriched.csv"), index=False)
    users_df.to_csv(os.path.join(OUTPUT_DIR, "users_enriched.csv"), index=False)
    subreddits_df.to_csv(os.path.join(OUTPUT_DIR, "subreddits_enriched.csv"), index=False)
    images_df.to_csv(os.path.join(OUTPUT_DIR, "images_enriched.csv"), index=False)
    print("Lưu xong!")


def _compute_local_fastrp(G, posts_df, emb_dim=64, seed=42):
    """
    Tính FastRP-style embedding cho Post nodes dựa trên 1-hop và 2-hop neighbors.
    Dùng sparse random projection để tránh tràn bộ nhớ với graph lớn.
    """
    rng = np.random.RandomState(seed)

    post_ids = [str(pid) for pid in posts_df["post_id"]]
    post_idx = {pid: i for i, pid in enumerate(post_ids)}
    n_posts  = len(post_ids)

    # Lấy tất cả neighbor IDs của các Post (1-hop)
    all_neighbor_ids = set()
    for pid in post_ids:
        if G.has_node(pid):
            all_neighbor_ids.update(G.neighbors(pid))
    all_neighbor_ids -= set(post_ids)     # loại Post-Post (không có CROSS_POST)
    neighbor_list = sorted(all_neighbor_ids)
    neighbor_idx  = {nid: i for i, nid in enumerate(neighbor_list)}
    n_neighbors   = len(neighbor_list)

    # Random projection matrix: neighbor_space → emb_dim
    R = rng.randn(n_neighbors, emb_dim).astype(np.float32) / np.sqrt(emb_dim)

    # 1-hop embedding: post → sum of neighbor projections
    emb_1hop = np.zeros((n_posts, emb_dim), dtype=np.float32)
    for pid in post_ids:
        if not G.has_node(pid):
            continue
        pi = post_idx[pid]
        nbrs = [n for n in G.neighbors(pid) if n in neighbor_idx]
        if nbrs:
            indices = [neighbor_idx[n] for n in nbrs]
            emb_1hop[pi] = R[indices].mean(axis=0)

    # 2-hop embedding: post → 1-hop embedding of neighbors that are also Posts
    # (không có Post-Post trực tiếp, nhưng Post có thể share User/Subreddit)
    emb_2hop = np.zeros((n_posts, emb_dim), dtype=np.float32)
    for pid in post_ids:
        if not G.has_node(pid):
            continue
        pi = post_idx[pid]
        two_hop_posts = set()
        for nbr in G.neighbors(pid):
            for nbr2 in G.neighbors(nbr):
                if nbr2 in post_idx and nbr2 != pid:
                    two_hop_posts.add(nbr2)
        if two_hop_posts:
            indices_2 = [post_idx[p2] for p2 in two_hop_posts]
            emb_2hop[pi] = emb_1hop[indices_2].mean(axis=0)

    # Kết hợp 1-hop và 2-hop (theo FastRP formulation)
    fastrp_arr = emb_1hop + 0.5 * emb_2hop

    # L2 normalize
    norms = np.linalg.norm(fastrp_arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    fastrp_arr = fastrp_arr / norms

    np.save(os.path.join(OUTPUT_DIR, "post_fastrp.npy"), fastrp_arr)
    print(f"Đã lưu local post_fastrp.npy: shape={fastrp_arr.shape}")


# ===================== MAIN =====================

def main():
    driver = check_neo4j_connection()
    if driver is not None:
        try:
            import_to_neo4j(driver)
        finally:
            driver.close()
    else:
        run_local_graph_fallback()


if __name__ == "__main__":
    main()
