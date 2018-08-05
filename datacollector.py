import os.path
import sqlite3

class DataCollector():
    def __init__(self, param_ranges, name="meta.db"):
        self.name = name
        self.connection = sqlite3.connect(name)
        self.cursor = self.connection.cursor()

        # Create the metatable if is does not exist
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metadata';")
        if not self.cursor.fetchone():
            columns = ["loss REAL"]
            for key in param_ranges.keys():
                if type(param_ranges[key][0]) == int:
                    columns.append(key + " INTEGER")
                else:
                    columns.append(key + " REAL")
            query = "CREATE TABLE metadata(" + ",".join(columns) + ");"
            self.cursor.execute(query)
            self.connection.commit()

        self.insert = "INSERT INTO metadata(loss," + ",".join(param_ranges.keys()) + ") VALUES (" + ",".join(["?"] * (len(param_ranges.keys()) + 1)) + ");"
        self.queries = []

    # save_params only collects the parameters, because it is much faster to 
    # commit everyting in one batch
    def save_params(self, params, loss):
        self.queries.append([loss] + list(params.values()))

    # Commit commits all the collected parameter tuples into the database in one batch
    def commit(self):
        self.cursor.executemany(self.insert, self.queries)
        self.connection.commit()
        self.queries = []
