from neo4j import GraphDatabase
import time

class InsertToNeo4J:
    def __init__(self):
        self.driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "12345678"))

    @staticmethod
    def delete_all(tx):
        tx.run("MATCH (n) detach delete (n)")

    @staticmethod
    def create_node_object(tx, name):
        tx.run("CREATE (:disease {name: $name})", name=name)

    @staticmethod
    def create_node(tx, name):
        tx.run("CREATE (:symptom {name: $name})", name=name)

    @staticmethod
    def create_relationship(tx, person1, person2, relation_type):
        tx.run(
            "MATCH (p1:symptom {name: $person1}), (p2:disease {name: $person2}) "
            "CREATE (p1)-[:`" + relation_type + "`]->(p2)",
            person1=person1, person2=person2
        )

    def main(self, df):

        with self.driver.session() as session:
            session.execute_write(self.delete_all)

        # print("DELETED")

        time.sleep(2)

        for each in df['Symptom'].unique():
            with self.driver.session() as session:
                session.execute_write(self.create_node, each)

        for each in df['Disease'].unique():
            with self.driver.session() as session:
                session.execute_write(self.create_node_object, each)

        df = df.drop_duplicates()
        for each in df.iterrows():
            with self.driver.session() as session:
                session.execute_write(self.create_relationship, each[1]['Symptom'], each[1]['Disease'], each[1]['Relation'])

        self.driver.close()
