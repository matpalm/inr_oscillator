import sqlite3
import numpy as np
from typing import List
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LossRow:
    idx: int
    loss: float
    huber: float
    stft: float


class SampleDB(object):

    def __init__(self, db_file=None, check_same_thread=True):
        if db_file is None:
            db_file = Path(__file__).with_name("sample.db")
        self.conn = sqlite3.connect(db_file, check_same_thread=check_same_thread)
        self.create_if_required()

    def create_if_required(self):
        c = self.conn.cursor()
        try:
            c.execute("""create table cv_values (
                            run text,
                            idx int,
                            cv0 real,
                            cv1 real,
                            cv2 real,
                            cv3 real,
                            captured bool,
                            primary key (run, idx)
                        )""")
            c.execute("create index idx_cv_values_run_idx on cv_values(run, idx)")
        except sqlite3.OperationalError:
            # assume table already exists? clumsy...
            pass
        try:
            c.execute("""create table losses (
                            run text,
                            idx int,
                            model text,
                            loss real,
                            huber real,
                            stft real,
                            primary key (run, idx, model)
                        )""")
            c.execute("create index idx_losses_run_model_idx on losses(run, model)")
        except sqlite3.OperationalError as e:
            # assume table already exists? clumsy...
            pass

    def set_cv_values(self, run: str, idx: int, cv: List[float]):
        run = str(run)
        c = self.conn.cursor()
        c.execute(
            """
            insert into cv_values
             (run, idx, cv0, cv1, cv2, cv3, captured)
             values (?, ?, ?, ?, ?, ?, 0)
            """,
            (run, idx, cv[0], cv[1], cv[2], cv[3]),
        )
        self.conn.commit()

    def set_cv_values_from_npy(self, run: str, cv_values: np.array):
        run = str(run)
        assert len(cv_values.shape) == 2
        assert cv_values.shape[-1] == 4
        c = self.conn.cursor()
        for idx, cv in enumerate(cv_values):
            c.execute(
                """
                insert into cv_values
                 (run, idx, cv0, cv1, cv2, cv3, captured)
                 values (?, ?, ?, ?, ?, ?, 0)
                """,
                (run, idx, cv[0], cv[1], cv[2], cv[3]),
            )
        self.conn.commit()

    def cv_values_for(self, run: str, idx: int = None):
        """
        Args:
            run: always needs to be set
            idx: if None, return all ( ordered by idx )
                 if a list, return entries as dict { entry: cv_values }
                 if int, return just that entry
        """

        run = str(run)
        c = self.conn.cursor()
        if idx is None:
            # all idxs; idx aligns with np row_id
            c.execute(
                """
                select cv0, cv1, cv2, cv3
                from cv_values
                where run=?
                order by idx
                """,
                (run,),
            )
            cvss = []
            for cvs in c.fetchall():
                cvss.append(cvs)
            return np.array(cvss)
        elif isinstance(idx, list):
            idxs = [int(i) for i in idx]
            placeholders = ",".join(["?"] * len(idxs))
            c.execute(
                f"""
                select idx, cv0, cv1, cv2, cv3
                from cv_values
                where run=? and idx in ({placeholders})
                """,
                (run, *idxs),
            )
            idx_to_row = {}
            result = []
            for i, (idx, cv0, cv1, cv2, cv3) in enumerate(c.fetchall()):
                idx_to_row[idx] = i
                result.append(np.array([cv0, cv1, cv2, cv3]))
            return idx_to_row, np.stack(result)
        else:
            c.execute(
                """
                select cv0, cv1, cv2, cv3
                from cv_values
                where run=? and idx=?
                order by idx
                """,
                (run, idx),
            )
            return np.array(c.fetchone())

    def captured_stats_for(self, run: str):
        run = str(run)
        c = self.conn.cursor()
        c.execute(
            """
            select captured, count(*) as c
            from cv_values
            where run=?
            group by captured
            """,
            (run,),
        )
        vals = c.fetchall()
        if vals == []:
            raise Exception(f"no entries for run=[{run}] ?")
        captured = {True: 0, False: 0}
        for cb, count in vals:
            captured[bool(cb)] += count
        return captured

    def idxs_to_capture(self, run: str):
        run = str(run)
        c = self.conn.cursor()
        c.execute(
            """
            select idx
            from cv_values
            where run=? and captured=0
            order by idx
            """,
            (run,),
        )
        return [r[0] for r in c.fetchall()]

    def set_captured(self, run: str, idx: int):
        run = str(run)
        c = self.conn.cursor()
        c.execute(
            """
            update cv_values
            set captured=1
            where run=? and idx=?
            """,
            (run, idx),
        )
        self.conn.commit()

    def set_losses(
        self, run: str, idx: int, model: str, loss: float, huber: float, stft: float
    ):
        run = str(run)
        c = self.conn.cursor()
        c.execute(
            """
            insert into losses
            (run, idx, model, loss, huber, stft)
            values (?, ?, ?, ?, ?, ?)
            on conflict(run, idx, model)
            do update set huber=excluded.huber,
                          stft=excluded.stft
            """,
            (run, idx, str(model), loss, huber, stft),
        )
        self.conn.commit()

    def losses_for(self, run: str, model: str):
        run = str(run)
        c = self.conn.cursor()
        c.execute(
            """
            select idx, loss, huber, stft
            from losses
            where run=? and model=?
            order by idx;
            """,
            (run, model),
        )
        return [LossRow(idx=i, loss=l, huber=h, stft=s) for i, l, h, s in c.fetchall()]

    def duplicate_run_with_idx_offset(
        self, src_run: str, dest_run: str, idx_offset: int
    ):
        """
        duplicate entries from src_run as dest_run
        but with dest_run.idx values offset by +idx_offset in cv_values
        """

        assert idx_offset >= 0
        src_run = str(src_run)
        dest_run = str(dest_run)
        c = self.conn.cursor()
        c.execute(
            """
            insert into cv_values
            (run, idx, cv0, cv1, cv2, cv3, captured)
                select ?, idx + ?, cv0, cv1, cv2, cv3, captured
                from cv_values
                where run=?
            """,
            (dest_run, idx_offset, src_run),
        )
        c.execute(
            """
            insert into losses
            (run, idx, model, loss, huber, stft)
                select ?, idx + ?, model, loss, huber, stft
                from losses
                where run=?
            """,
            (dest_run, idx_offset, src_run),
        )
        self.conn.commit()

    def delete_run(self, run: str):
        c = self.conn.cursor()
        c.execute("delete from cv_values where run=?", (run,))
        c.execute("delete from losses where run=?", (run,))
        self.conn.commit()

    def dump_stats(self):
        c = self.conn.cursor()
        c.execute("""
            select run, model, count(*) as c
            from losses
            group by run, model
            """)
        print("losses")
        print("\n".join(map(str, c.fetchall())))
        print()
        c.execute("""
            select run, count(*) as c
            from cv_values
            group by run order by run
            """)
        print("cv_values")
        print("\n".join(map(str, c.fetchall())))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--run", type=str, nargs="+")
    parser.add_argument("--delete", action="store_true")
    opts = parser.parse_args()

    db = SampleDB()

    if opts.delete:
        for run in opts.run:
            db.delete_run(run)
        exit()

    db.dump_stats()

    # always do stats
