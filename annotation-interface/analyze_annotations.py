import pandas as pd
from sklearn.metrics import ndcg_score
import random
from scipy import stats
import numpy as np
from ast import literal_eval
from scipy.optimize import minimize
from itertools import combinations
import krippendorff

# taken from https://github.com/webis-de/acl20-efficient-argument-quality-annotation
class BradleyTerry:
    def __init__(self, comparisons, parsefunc=None):
        """
        Constructor
        :param comparisons: list of comparisons
        :param parsefunc: optionally pass a custom parsign function to cope with different data formats
        """
        parsefunc = parsefunc if parsefunc is not None else self.__parsefunc__
        self.items, self.comparisons, self.merits = parsefunc(comparisons)

    @staticmethod
    def __parsefunc__(comparisons) -> tuple:
        """
        Function to parse supplied comparison data to the format needed by the model
        :param comparisons: comparison data
        :return
        """
        items = list(set([x[0] for x in comparisons]+[x[1] for x in comparisons]))

        # Mapping
        items_parsed = {x: i for i, x in enumerate(items)}

        # Mapped comparisons
        comparisons_parsed = []
        for arg1_id, arg2_id, tie in comparisons:
            comparisons_parsed.append([
                items_parsed[arg1_id],
                items_parsed[arg2_id],
                tie
            ])

        # Initialize zero-vector for merits
        merits = np.zeros(len(items))

        return (items_parsed, comparisons_parsed, merits)

    @staticmethod
    def __pfunc__(i: float, j: float, t: float) -> float:
        """
        Function to compute pairwise comparison probabilities of non-ties
        :param i: merit of the winning item
        :param j: merit of the loosing item
        :param s: annotation quality score
        :param t: difference threshold
        :return: propability of item i beating item j
        """
        p = np.exp(i) / (np.exp(i) + np.exp(j) * np.exp(t))
        return np.log10(p)

    @staticmethod
    def __tfunc__(i: float, j: float, t: float) -> float:
        """
        Function to compute pairwise comparison probabilities of ties
        :param i: merit of the winning item
        :param j: merit of the loosing item
        :param t: difference threshold
        :return: propability of item i beating item j
        """
        f1 = np.exp(i) * np.exp(j) * (np.square(np.exp(t)) - 1)
        f2 = (np.exp(i) + np.exp(j) * np.exp(t)) * (np.exp(i) * np.exp(t) + np.exp(j))
        p = f1 / f2
        return np.log10(p)

    def __rfunc__(self, i: float, l: float) -> float:
        """
        Function to compute regularized probability
        :param i: item merit
        :param l: regularization factor
        :return: value of __pfunc__ for matches with dummy item weighted by l
        """
        return l * (self.__pfunc__(i, 1, 0) + self.__pfunc__(1, i, 0))

    def __log_likelihood__(self, merits: np.ndarray) -> float:
        """
        Log-Likelihood Function
        :param merits: merit vector
        :return: log-likelihood value
        """
        k: float = 0  # Maximization sum

        # Summing Edge Probabilities
        for arg1, arg2, tie in self.comparisons:
            if tie:
                k += self.__tfunc__(merits[arg1], merits[arg2], self.threshold)
            else:
                k += self.__pfunc__(merits[arg1], merits[arg2], self.threshold)

        # Regularization
        for x in range(len(self.items)):
            k += self.__rfunc__(merits[x], self.regularization)

        return -1 * k

    def fit(self, regularization: float = 0, threshold: float = 0) -> None:
        """
        Optimize the model for merits
        :param regularization: regularization parameter
        :param threshold: difference threshold
        """
        self.merits = np.ones(len(self.items))
        self.threshold = threshold
        self.regularization = regularization

        res = minimize(self.__log_likelihood__, self.merits, method='BFGS', options={"maxiter": 100})
        self.merits = res.x

    def get_merits(self, normalize=False) -> list:
        """
        Returns the merits mapped to items
        :param normalize: if true, returns normalized merit vector to 0-1 range instead of original scores
        :return: dict in the form of {argument_id: merit} sorted by merits
        :exception: Exception if model was not fitted
        """
        if not self.merits.any():
            raise Exception('Model has to be fitted first!')
        else:
            d = {argument_id: self.merits[index] for argument_id, index in self.items.items()}
            if normalize:
                mi = min(d.values())
                ma = max(d.values())
                def normalize(mi, ma, v): return (v-mi)/(ma-mi)
                d.update({k: normalize(mi, ma, v) for k, v in d.items()})
            return sorted(d.items(), key=lambda kv: kv[1])


class PairwiseAggregator:
    def __init__(self, threshold, margin, log_scores=True, logit_scores=False):
        self.threshold = threshold
        self.margin = margin
        self.log_scores = log_scores
        self.logit_scores = logit_scores

    def _infer_tie(self, p):
        if not 0 <= p <= 1:
            raise ValueError('Got invalid p of ' + str(p) + '. Expected p in Interval [0, 1]')

        if not 0 <= (self.threshold + self.margin) <= 1:
            raise ValueError('Got invalid threshold and margin of ' + str(self.threshold + self.margin) +
                             '. Expected p in Interval [0, 1]')

        if not 0 <= (self.threshold - self.margin) <= 1:
            raise ValueError('Got invalid threshold and margin of ' + str(self.threshold - self.margin) +
                             '. Expected p in Interval [0, 1]')

        if p > self.threshold + self.margin:
            return False
        elif p < self.threshold - self.margin:
            return False
        else:
            return True

    def _order_pair(self, id_a, id_b, p):
        if not 0 <= p <= 1:
            raise ValueError('Got invalid p of ' + str(p) + '. Expected p in Interval [0, 1]')
        if p >= self.threshold:
            return id_a, id_b, p
        else:
            return id_b, id_a, 1 - p

    def __call__(self, pairwise_scores: pd.DataFrame) -> pd.DataFrame:
        if self.log_scores:
           pairwise_scores["score"] = pairwise_scores["score"].apply(np.exp)
        if self.logit_scores:
            pairwise_scores["score"] = pairwise_scores["score"].apply(lambda x: np.exp(x)/(1+np.exp(x)))
        return pairwise_scores

        """"
        data = []
        if self.log_scores:
            for _, (id_a, id_b, p) in pairwise_scores.iterrows():
                data.append(self._order_pair(id_a, id_b, np.exp(p)))
        else:
            for _, (id_a, id_b, p) in pairwise_scores.iterrows():
                data.append(self._order_pair(id_a, id_b,p))
        return pd.DataFrame(data, columns=pairwise_scores.columns)
        """


class BradleyTerryAggregator(PairwiseAggregator):
    def __init__(self, tie_margin: float = 0.05, tie_threshold: float = 0.05, regularization: float = 0.2,
                 max_iter: int = 100, log_scores=False, logit_scores=False, normalize_scores=False, cython=False):
        """
        Constructor
        :param tie_margin: score margin to declare ties
        :param regularization: regularization parameter
        :param tie_threshold: difference threshold
        :param max_iter: maximum iterations for the LL optimizer
        """
        super().__init__(0.5, tie_margin, log_scores=log_scores, logit_scores=logit_scores)
        self.threshold = tie_threshold
        self.regularization = regularization
        self.max_iter = max_iter
        self.normalize = normalize_scores

        if not cython:
            self.optimize = self._optimize_python
        else:
            self.optimize = self._optimize_cython

    def __str__(self):
        return "bradleyterry"

    def _optimize_cython(self, comparisons, n_samples, regularization, threshold):
        from ._bradleyterry import __log_likelihood__
        # Transform comparisons into fast iterable matrix
        comparison_matrix = np.zeros(shape=(len(comparisons), 3), dtype=np.intc)
        for i, (id_a, id_b, tie) in enumerate(comparisons):
            comparison_matrix[i, 0] = int(id_a)
            comparison_matrix[i, 1] = int(id_b)
            comparison_matrix[i, 2] = int(tie)
        # Initialize merit vector
        merits = np.ones(shape=(n_samples,), dtype=np.double)
        # Optimize using BFGS
        res = minimize(__log_likelihood__, merits, (comparison_matrix, regularization, threshold), method="BFGS")
        return res.x

    def _optimize_python(self, comparisons, n_samples, regularization, threshold):
        # Initialize merit vector
        merits = np.ones(shape=(n_samples,), dtype=np.double)
        # Optimize using BFGS
        res = minimize(self.__log_likelihood__, merits, (comparisons, regularization, threshold), method="BFGS")
        return res.x

    @staticmethod
    def __pfunc__(i: float, j: float, t: float) -> float:
        """
        Function to compute pairwise comparison probabilities of non-ties
        :param i: merit of the winning item
        :param j: merit of the loosing item
        :param t: difference threshold
        :return: probability of item i beating item j
        """
        p = np.exp(i) / (np.exp(i) + np.exp(j) * np.exp(t))
        return np.log10(p)

    @staticmethod
    def __tfunc__(i: float, j: float, t: float) -> float:
        """
        Function to compute pairwise comparison probabilities of ties
        :param i: merit of the winning item
        :param j: merit of the loosing item
        :param t: difference threshold
        :return: probability of item i beating item j
        """
        f1 = np.exp(i) * np.exp(j) * (np.square(np.exp(t)) - 1)
        f2 = (np.exp(i) + np.exp(j) * np.exp(t)) * (np.exp(i) * np.exp(t) + np.exp(j))
        p = f1 / f2
        return np.log10(p)

    def __rfunc__(self, i: float, l: float) -> float:
        """
        Function to compute regularized probability
        :param i: item merit
        :param l: regularization factor
        :return: value of __pfunc__ for matches with dummy item weighted by l
        """
        return l * (self.__pfunc__(i, 1, 0) + self.__pfunc__(1, i, 0))

    def __log_likelihood__(self, merits: np.ndarray, comparisons: np.ndarray, regularization: float, threshold: float) -> float:
        """
        Log-Likelihood Function
        :param merits: merit vector
        :return: log-likelihood value
        """
        k: float = 0  # Maximization sum
        # Summing Edge Probabilities
        for arg1, arg2, tie in comparisons:
            if tie:
                k += self.__tfunc__(merits[arg1], merits[arg2], threshold)
            else:
                k += self.__pfunc__(merits[arg1], merits[arg2], threshold)
        # Regularization
        for i in range(merits.shape[0]):
            k += self.__rfunc__(merits[i], regularization)
        return -1 * k

    def __call__(self, pairwise_scores: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the aggregation and return the calculated merits.
        :param pairwise_scores: pairwise score data
        """

        pairwise_scores = super().__call__(pairwise_scores)
        self.items = list(set(pairwise_scores["id_a"].unique().tolist() + pairwise_scores["id_b"].unique().tolist()))
        item_mapping = {x: i for i, x in enumerate(self.items)}

        # Mapped comparisons
        self.comparisons = []
        for _, (id_a, id_b, p) in pairwise_scores.iterrows():
            tie = super()._infer_tie(p)
            self.comparisons.append([
                item_mapping[id_a],
                item_mapping[id_b],
                tie
            ])
        res = self.optimize(comparisons=self.comparisons, n_samples=len(self.items), regularization=self.regularization,
                            threshold=self.threshold)

        scores = {doc_id: res[index] for doc_id, index in item_mapping.items()}
        df = pd.DataFrame(scores.items(), columns=["docno", "score"])
        if self.normalize:
            df["score"] = (df["score"] - df["score"].min()) / (df["score"].max() - df["score"].min())
            return df
        else:
            return df


class GreedyAggregator(PairwiseAggregator):
    """

    References
    ----------
    William W. Cohen, Robert E. Schapire, Yoram Singer Learning to Order Things. J. Artif. Intell. Res. 10: 243-270 (1999)

    """

    def __init__(self, log_scores: bool = True, logit_scores: bool = False):
        super().__init__(0.5, 0, log_scores=log_scores, logit_scores=logit_scores)

    def __str__(self):
        return "greedy"

    def __call__(self, pairwise_scores: pd.DataFrame) -> pd.DataFrame:
        pairwise_scores = super().__call__(pairwise_scores)

        items = list(set(pairwise_scores["id_a"].unique().tolist() + pairwise_scores["id_b"].unique().tolist()))
        item_mapping = {x: i for i, x in enumerate(items)}

        # Construct score lookup
        scores = np.zeros(shape=(len(items), len(items)), dtype=np.float32)
        for _, (id_a, id_b, p) in pairwise_scores.iterrows():
            scores[item_mapping[id_a], item_mapping[id_b]] = p

        # Calculate initial score for each item
        pi_v = np.zeros(shape=len(items), dtype=np.float32)
        for v in range(pi_v.shape[0]):
            pi_v[v] = np.sum(scores[v, :]) - np.sum(scores[:, v])

        # Initialize ranks
        ranks = np.zeros(shape=len(items), dtype=np.int32)

        for i in range(ranks.shape[0]):
            # Choose remaining item with the highest potential
            t = np.argmax(np.where(ranks == 0, pi_v, -np.inf))
            # Assign rank to t (inverted ranks to qualify as descendingly sortable scores)
            ranks[t] = len(ranks) - i
            # Adjust remaining scores
            for v in np.where(ranks == 0)[0]:
                pi_v[v] = pi_v[v] + scores[t, v] - scores[v, t]

        return pd.DataFrame(zip(item_mapping.keys(), ranks), columns=["docno", "score"])


VALID_MERIT = None
def compute_ranking(df, ids):
    ranking = {}
    nan_count = 0
    global VALID_MERIT
    for id_source in ids:
        df_source = df[df['id_source'] == id_source]
        #print(df_source) 
        comparisons = []
        #for i, row in df_source.iterrows():
        #    if row['slider_score'] == 0:
        #        comparisons.append((row['id_model1'], row['id_model2'], 1))
        #        comparisons.append((row['id_model2'], row['id_model1'], 0))
        #    elif row['slider_score'] == 1:
        #        comparisons.append((row['id_model1'], row['id_model2'], 0.875))
        #        comparisons.append((row['id_model2'], row['id_model1'], 0.125))
        #    elif row['slider_score'] == 2:
        #        comparisons.append((row['id_model1'], row['id_model2'], 0.75))
        #        comparisons.append((row['id_model2'], row['id_model1'], 0.25))
        #    elif row['slider_score'] == 3:
        #        comparisons.append((row['id_model1'], row['id_model2'], 0.6125))
        #        comparisons.append((row['id_model2'], row['id_model1'], 0.3875))
        #    elif row['slider_score'] == 4:
        #        comparisons.append((row['id_model1'], row['id_model2'], 0.5))
        #        comparisons.append((row['id_model2'], row['id_model1'], 0.5))
        #    elif row['slider_score'] == 5:
        #        comparisons.append((row['id_model1'], row['id_model2'], 0.375))
        #        comparisons.append((row['id_model2'], row['id_model1'], 0.625))
        #    elif row['slider_score'] == 6:
        #        comparisons.append((row['id_model1'], row['id_model2'], 0.25))
        #        comparisons.append((row['id_model2'], row['id_model1'], 0.75))
        #    elif row['slider_score'] == 7:
        #        comparisons.append((row['id_model1'], row['id_model2'], 0.125))
        #        comparisons.append((row['id_model2'], row['id_model1'], 0.875))
        #    elif row['slider_score'] == 8:
        #        comparisons.append((row['id_model1'], row['id_model2'], 0))
        #        comparisons.append((row['id_model2'], row['id_model1'], 1))
        #    else:
        #        print('ERROR IN SLIDER SCORE')



        for i, row in df_source.iterrows():
            if row['slider_score'] == 4:
                comparisons.append((row['id_model1'], row['id_model2'], True))
            elif row['slider_score'] < 4:
                comparisons.append((row['id_model1'], row['id_model2'], False))
            elif row['slider_score'] > 4:
                comparisons.append((row['id_model2'], row['id_model1'], False))
            else:
                print('ERROR IN SLIDER SCORE')

        # avearge scores of same comparisons
        #comparisons = pd.DataFrame(comparisons, columns=['id_a', 'id_b', 'score'])
        #comparisons = comparisons.groupby(['id_a', 'id_b']).mean().reset_index().values.tolist()

        # Initialize the model with given comparisons
        bt = BradleyTerry(comparisons)

        #comparison_df = pd.DataFrame(comparisons, columns=['id_a', 'id_b', 'score'])
        #print(comparison_df)
        #bt2 = GreedyAggregator()(comparison_df)
        #print(bt2)
        # to list of tuples
        #merits = [(x[0], x[1]) for x in bt2.values.tolist()]
        #print(merits)

        # Fit the model using supplied hyperparameters
        bt.fit(regularization=0.3, threshold=0.01)

        merits = bt.get_merits(normalize=True)
        #print(merits)
        if np.isnan(merits[0][1]):
            nan_count += 1
            merits = VALID_MERIT
            # shuffle second values in merits
            second_values = [x[1] for x in merits]
            random.shuffle(second_values)
            merits = [(merits[i][0], second_values[i]) for i in range(len(merits))]
        for i, merit in enumerate(merits):
            if merit[0] not in ranking:
                ranking[merit[0]] = []
            ranking[merit[0]].append(merit[1])
        VALID_MERIT = merits

    if nan_count > 0:
        print('% nan count: ', nan_count/len(ids))
    return ranking


def compare_sampling(df, mode=None, annotators=[3,4,5,6,7], num_random=None):
    tmp_df = df[df['user_id'].isin(annotators)]
    ids = tmp_df.id_source.unique()
    full_ranking = compute_ranking(tmp_df, ids)

    l = [0, 1, 2, 3, 4, 5]
    n = len(l)
    circle_ids = [(l[i], l[(i+1) % n]) for i in range(n)]
    if mode == 'extended':
        circle_ids += [(0,3), (1,4), (2,5)]
    if mode == 'reduced':
        circle_ids = circle_ids[:-1]
    if mode == 'random':
        #create num_random combinations of ids
        circle_ids = []
        while len(circle_ids) < num_random:
            circle = random.sample(l, 2)
            if circle not in circle_ids:
                circle_ids.append(circle)
        print(circle_ids)

    models = list(set(tmp_df.id_model1.unique().tolist()+tmp_df.id_model2.unique().tolist()))
    random.shuffle(models)
    model_pos_in_circle = {model: i for i, model in enumerate(models)}
    # only keep rows in tmp_df where both models are in next to each other in the circle
    circle_tmp_df = tmp_df[tmp_df.apply(lambda x: (model_pos_in_circle[x['id_model1']],
                            model_pos_in_circle[x['id_model2']]) in circle_ids or (model_pos_in_circle[x['id_model2']], model_pos_in_circle[x['id_model1']]) in circle_ids, axis=1)]
    circle_ranking = compute_ranking(circle_tmp_df, ids)

    pearson_rs = []
    ndcg_1s = []
    for i in range(len(ids)):
        x = []
        y = []
        for key, value in full_ranking.items():
            x.append(value[i])
            y.append(circle_ranking[key][i])

        res = stats.pearsonr(x, y)
        ndcg_1 = ndcg_score([x], [y], k=1)
        pearson_rs.append(res.statistic)
        ndcg_1s.append(ndcg_1)
    rng = np.random.default_rng()
    res = stats.bootstrap((pearson_rs, ), np.mean, confidence_level=0.95, random_state=1, method='percentile', n_resamples=10000).confidence_interval
    return np.mean(pearson_rs), res.low, res.high, len(circle_tmp_df), np.mean(ndcg_1s)


def get_final_ranking(df):
    ranking = {}
    for _ in range(10):
        tmp_ranking = compute_ranking(df, df.id_source.unique())
        for k, v in tmp_ranking.items():
            if k not in ranking:
                ranking[k] = []
            ranking[k].append(v)
    mean_ranking = {}
    for k, v in ranking.items():
        mean_ranking[k] = np.mean(v)


    abs_counts = {}
    for id_source in df.id_source.unique():
        tmp_ranking = compute_ranking(df[df['id_source'] == id_source], [id_source])
        #unpack all values
        tmp_ranking = {k: v[0] for k, v in tmp_ranking.items()}
        # replace values with ranking
        tmp_ranking = {k: sorted(tmp_ranking, key=lambda x: tmp_ranking[x]).index(k)+1 for k in tmp_ranking}
        for k, v in tmp_ranking.items():
            if k not in abs_counts:
                abs_counts[k] = {1: 0, 2: 0, 3: 0, 4: 0}
            abs_counts[k][5-v] += 1
    print(abs_counts)
    # print mean abs_count for each model
    for k, v in abs_counts.items():
        print(k, np.sum([k2*v2 for k2, v2 in v.items()])/len(df.id_source.unique()))

    for i, x in enumerate(sorted(mean_ranking.items(), key=lambda kv: kv[1])[::-1]):
        print('Rank {}: {} (mean merit: {})'.format(i+1, x[0], x[1]))

    # print absolute counts each model won a comparisons
    counts = {}
    #df = df[(df['id_model1'].isin(['10a-00ss','60a-40ss'])) & (df['id_model2'].isin(['10a-00ss','60a-40ss']))]
    for i, row in df.iterrows():
        if row['slider_score'] == 4:
            if 'tie' not in counts:
                counts['tie'] = 0
            counts['tie'] += 1
        elif row['slider_score'] < 4:
            if row['id_model1'] not in counts:
                counts[row['id_model1']] = 0
            counts[row['id_model1']] += 1
        elif row['slider_score'] > 4:
            if row['id_model2'] not in counts:
                counts[row['id_model2']] = 0
            counts[row['id_model2']] += 1
        else:
            print('ERROR IN SLIDER SCORE')
    print('Absolute counts:')
    print(sorted(counts.items(), key=lambda kv: kv[1])[::-1])


def get_agreement(df):
    pearsons = []
    for _ in range(10):
        for user_id1 in df.user_id.unique():
            for user_id2 in df.user_id.unique():
                if user_id1 < user_id2:
                    tmp_pearsons = []
                    rankings1 = compute_ranking(df[df['user_id'] == user_id1], df.id_source.unique())
                    rankings2 = compute_ranking(df[df['user_id'] == user_id2], df.id_source.unique())
                    for i in range(len(list(rankings1.values())[0])):
                        x = []
                        y = []
                        for k in list(set(list(rankings1.keys())+list(rankings2.keys()))):
                            x.append(rankings1[k][i])
                            y.append(rankings2[k][i])
                        res = stats.pearsonr(x, y)
                        pearsons.append(res.statistic)
                        tmp_pearsons.append(res.statistic)
                    #print('User {} vs User {}: {}'.format(user_id1, user_id2, np.mean(tmp_pearsons)))
    pearsons = sum(pearsons)/len(pearsons)
    print('Pearson\'s R: {}'.format(pearsons))


def extract_post_id_parts(post_id):
    """Extract source_id, model1, and model2 from post_id with model names that contain underscores."""
    known_models = ['v11_only_lm', 'v11_r11_ppo', 'ppo_50a_50ss', 'base_model']

    # Find all models in the post_id with their positions
    found_models = []
    for model in known_models:
        if model in post_id:
            idx = post_id.find(model)
            found_models.append((idx, model))

    # Sort by position
    found_models.sort(key=lambda x: x[0])

    if len(found_models) != 2:
        raise ValueError(f"Expected 2 models in {post_id}, found {len(found_models)}")

    model1 = found_models[0][1]
    model2 = found_models[1][1]

    # Extract source_id (everything before first model minus the underscore)
    source_id = post_id[:found_models[0][0] - 1]

    return source_id, model1, model2


def analyze_prestudy():
    columns = ['a_id', 'user_id', 'post_id', 'post_text', 'annotation_date', 'result', 'issue', 'comments']
    df = pd.read_csv('./study_pairs_rel_results.csv')
    print(len(df))
    df['result'] = df['result'].apply(literal_eval)
    df['slider_score'] = df['result'].apply(lambda x: int(x['rangeslider1']))

    # Extract id_source, id_model1, id_model2 properly handling model names with underscores
    post_id_parts = df['post_id'].apply(extract_post_id_parts)
    df['id_source'] = post_id_parts.apply(lambda x: x[0])
    df['id_model1'] = post_id_parts.apply(lambda x: x[1])
    df['id_model2'] = post_id_parts.apply(lambda x: x[2])

    # keep only users that are in [35, 36, 37]
    df = df[df['user_id'].isin([35,36,37])]
    print(f'Total annotations: {len(df)}')
    print('Annotations per user:')
    print(df['user_id'].value_counts().sort_index())

    # only keep post_ids that appear 3 times in the df (once per user: 35, 36, 37)
    ids_to_keep = df['post_id'].value_counts()[df['post_id'].value_counts() == 3].index.tolist()
    df = df[df['post_id'].isin(ids_to_keep)]
    print(f'\nAfter filtering for common post_ids: {len(df)}')
    print('Annotations per user after filtering:')
    print(df['user_id'].value_counts().sort_index())

    print('\nCalculating agreement')
    get_agreement(df)
    print('Calculating final ranking')
    get_final_ranking(df)


def analyze_study():
    columns = ['a_id', 'user_id', 'post_id', 'post_text', 'annotation_date', 'result', 'issue', 'comments']
    df_pre = pd.read_csv('../../data/style-transfer/prestudy_pairs_results.csv')
    df = pd.read_csv('../../data/style-transfer/study_pairs_results.csv')
    print(len(df))
    df = pd.concat([df_pre, df])
    df = df[~df['post_id'].str.contains('_45a-55ss')]
    df['result'] = df['result'].apply(literal_eval)
    df['slider_score'] = df['result'].apply(lambda x: int(x['rangeslider1']))
    df['id_source'] = df['post_id'].apply(lambda x: x.split('_')[0])
    df['id_model1'] = df['post_id'].apply(lambda x: x.split('_')[1])
    df['id_model2'] = df['post_id'].apply(lambda x: x.split('_')[-1])

    # keep only users that are in [3,4,5,6,7]
    df = df[df['user_id'].isin([3,4,5,6,7])]
    print(len(df))

    # only keep post_ids that appear 5 times in the df
    ids_to_keep = df['post_id'].value_counts()[df['post_id'].value_counts() == 5].index.tolist()
    df = df[df['post_id'].isin(ids_to_keep)]

    print('Calculating agreement')
    get_agreement(df)
    print('Calculating final ranking')
    get_final_ranking(df)


def create_latex_diff_original(source_text, inappropriate_part, rewritten_part):
    """
    Create LaTeX formatted diff for ORIGINAL view (like the interface).
    Shows inappropriate part with underline, bold deletions inside.

    Args:
        source_text: The original sentence
        inappropriate_part: The part to be edited
        rewritten_part: The replacement text

    Returns:
        LaTeX formatted string showing original with highlighted edits
    """
    import diff_match_patch as dmp_module

    if not inappropriate_part or inappropriate_part in ('nan', 'None', ''):
        return latex_escape(source_text)

    if not rewritten_part or rewritten_part in ('nan', 'None'):
        rewritten_part = ''

    if inappropriate_part not in source_text:
        return latex_escape(source_text)

    dmp = dmp_module.diff_match_patch()
    diffs = dmp.diff_main(inappropriate_part, rewritten_part)
    dmp.diff_cleanupSemantic(diffs)

    before, _, after = source_text.partition(inappropriate_part)

    # Build the highlighted inappropriate part (with bold red deletions)
    highlighted_inappropriate = ''
    for op, data in diffs:
        text = latex_escape(data)
        if op == dmp.DIFF_DELETE:
            # Bold red for deletions (what will be removed)
            highlighted_inappropriate += r'\textcolor{red}{\textbf{' + text + '}}'
        elif op == dmp.DIFF_EQUAL:
            # Regular text for parts that stay
            highlighted_inappropriate += text
        # DIFF_INSERT is ignored in original view

    # Red underline the entire inappropriate part
    result = (
        latex_escape(before) +
        r'\textcolor{red}{\uline{' + highlighted_inappropriate + '}}' +
        latex_escape(after)
    )
    return result


def create_latex_diff_edited(source_text, inappropriate_part, rewritten_part):
    """
    Create LaTeX formatted diff for EDITED view (like the interface).
    Shows rewritten part with underline, bold insertions inside.

    Args:
        source_text: The original sentence
        inappropriate_part: The part to be edited
        rewritten_part: The replacement text

    Returns:
        LaTeX formatted string showing edited version with highlighted changes
    """
    import diff_match_patch as dmp_module

    if not inappropriate_part or inappropriate_part in ('nan', 'None', ''):
        return latex_escape(source_text)

    if not rewritten_part or rewritten_part in ('nan', 'None'):
        rewritten_part = ''

    if inappropriate_part not in source_text:
        return latex_escape(source_text)

    dmp = dmp_module.diff_match_patch()
    diffs = dmp.diff_main(inappropriate_part, rewritten_part)
    dmp.diff_cleanupSemantic(diffs)

    before, _, after = source_text.partition(inappropriate_part)

    # Build the highlighted rewritten part (with bold green insertions)
    highlighted_rewritten = ''
    for op, data in diffs:
        text = latex_escape(data)
        if op == dmp.DIFF_INSERT:
            # Bold green for insertions (what was added)
            highlighted_rewritten += r'\textcolor{green}{\textbf{' + text + '}}'
        elif op == dmp.DIFF_EQUAL:
            # Regular text for parts that stayed the same
            highlighted_rewritten += text
        # DIFF_DELETE is ignored in edited view

    # Green underline the entire rewritten part
    result = (
        latex_escape(before) +
        r'\textcolor{green}{\uline{' + highlighted_rewritten + '}}' +
        latex_escape(after)
    )
    return result


def latex_escape(text):
    """Escape LaTeX special characters."""
    if not text:
        return ''
    replacements = {
        '\\': r'\textbackslash{}',
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def get_agreement_examples_lowest_scores(df):
    """
    Get 5 bad examples from each model for each category.

    For nat, sim, flu: Find examples with lowest sum in target category that have highest difference with other categories
    For hl: Find examples with low sum in ALL categories
    """
    categories = ['nat', 'sim', 'flu', 'hl']
    category_names = {
        'nat': 'Naturalness',
        'sim': 'Similarity',
        'flu': 'Fluency',
        'hl': 'Human-likeness'
    }

    # Filter to only keep post_ids that appear for all three users (3, 4, 5)
    df_filtered = df[df['user_id'].isin([3,4,5])]
    post_ids_per_user = df_filtered.groupby('user_id')['post_id'].apply(set)
    common_post_ids = set.intersection(*[s for s in post_ids_per_user.values])
    df_filtered = df_filtered[df_filtered['post_id'].isin(common_post_ids)]

    # Group by post_id and sum scores across annotators for each category
    score_sums = df_filtered.groupby('post_id')[categories].sum().reset_index()

    # Merge with model info
    model_info = df_filtered.groupby('post_id')[['id_model', 'post_text']].first().reset_index()
    score_sums = score_sums.merge(model_info, on='post_id')

    # Get list of all models
    all_models = sorted(score_sums['id_model'].unique())

    results = {}  # results[category][model] = list of examples

    for category in categories:
        results[category] = {}

        # Calculate quality metrics for all candidates
        candidates_by_model = {model: [] for model in all_models}

        for _, row in score_sums.iterrows():
            post_id = row['post_id']
            target_sum = row[category]
            model = row['id_model']

            # Calculate sum for other categories
            other_categories = [c for c in categories if c != category]
            other_sums = {c: row[c] for c in other_categories}

            if category == 'hl':
                # For human-likeness: we want low scores in ALL categories
                # Quality metric: negative total sum (lower total is better)
                total_sum = target_sum + sum(other_sums.values())
                quality_metric = -total_sum
            else:
                # For nat, sim, flu: we want low score in target but high in others
                # Quality metric: difference between mean of others and target
                mean_others = sum(other_sums.values()) / len(other_sums)
                quality_metric = mean_others - target_sum

            candidates_by_model[model].append({
                'post_id': post_id,
                'target_sum': target_sum,
                'other_sums': other_sums,
                'quality_metric': quality_metric,
                'id_model': model,
                'post_text': row['post_text']
            })

        # For each model, select top 5 examples by quality metric
        for model in all_models:
            model_candidates = candidates_by_model[model]
            # Sort by quality metric (descending - higher is better)
            model_candidates.sort(key=lambda x: -x['quality_metric'])
            # Take top 5
            results[category][model] = model_candidates[:5]

    # Print results
    print('\n' + '='*80)
    print('5 BAD EXAMPLES FROM EACH MODEL FOR EACH CATEGORY')
    print('='*80)

    for category in categories:
        print(f'\n{category_names[category]} ({category}):')
        if category == 'hl':
            print('  (Finding examples with LOW sum in ALL categories)')
        else:
            print(f'  (Finding examples with LOW sum in {category} but HIGH sum in others)')
        print('='*80)

        for model in all_models:
            examples = results[category][model]
            print(f'\n  Model: {model}')
            print('  ' + '-'*76)
            if len(examples) == 0:
                print('    No examples found.')
            else:
                for i, example in enumerate(examples, 1):
                    print(f'    Example {i}:')
                    print(f'      Post ID: {example["post_id"]}')
                    print(f'      {category.upper()} Sum: {example["target_sum"]:.1f}')
                    other_scores_str = ', '.join([f'{cat.upper()}: {score:.1f}' for cat, score in example['other_sums'].items()])
                    print(f'      Other Sums: {other_scores_str}')
                    if category == 'hl':
                        total = example["target_sum"] + sum(example['other_sums'].values())
                        print(f'      Total Sum: {total:.1f}')
                    else:
                        print(f'      Quality: {example["quality_metric"]:.2f}')

    print('\n' + '='*80)
    return results


def get_good_edit_examples(df):
    """
    Get 5 examples with high scores across ALL categories (good edits).
    Progressive relaxation to find examples with the best overall quality.
    """
    from collections import Counter

    categories = ['nat', 'sim', 'flu', 'hl']
    category_names = {
        'nat': 'Naturalness',
        'sim': 'Similarity',
        'flu': 'Fluency',
        'hl': 'Human-likeness'
    }

    # Filter to only keep post_ids that appear for all three users (3, 4, 5)
    df_filtered = df[df['user_id'].isin([3,4,5])]
    post_ids_per_user = df_filtered.groupby('user_id')['post_id'].apply(set)
    common_post_ids = set.intersection(*[s for s in post_ids_per_user.values])
    df_filtered = df_filtered[df_filtered['post_id'].isin(common_post_ids)]

    all_candidates = []

    # Progressive relaxation tiers for good examples
    tiers = [
        # Tier 1: 2/3 majority on all, all 4 categories must be excellent (5)
        {'min_excellent': 4, 'use_avg': False, 'min_score': 5, 'name': 'Tier 1: All categories = 5'},
        # Tier 2: 2/3 majority on all, all 4 categories must be good (4-5)
        {'min_excellent': 4, 'use_avg': False, 'min_score': 4, 'name': 'Tier 2: All categories >= 4'},
        # Tier 3: 2/3 majority on all, at least 3 categories must be good (4-5)
        {'min_excellent': 3, 'use_avg': False, 'min_score': 4, 'name': 'Tier 3: 3/4 categories >= 4'},
        # Tier 4: Avg scores, all 4 categories average >= 4
        {'min_excellent': 4, 'use_avg': True, 'min_score': 4, 'name': 'Tier 4: Avg all categories >= 4'},
        # Tier 5: Avg scores, at least 3 categories average >= 4
        {'min_excellent': 3, 'use_avg': True, 'min_score': 4, 'name': 'Tier 5: Avg 3/4 categories >= 4'},
        # Tier 6: Avg scores, all 4 categories average >= 3
        {'min_excellent': 4, 'use_avg': True, 'min_score': 3, 'name': 'Tier 6: Avg all categories >= 3'},
    ]

    for tier in tiers:
        if len(all_candidates) >= 5:
            break

        tier_candidates = []

        for post_id in df_filtered['post_id'].unique():
            post_data = df_filtered[df_filtered['post_id'] == post_id]

            scores_dict = {}

            if tier['use_avg']:
                # Use average scores
                for cat in categories:
                    scores_dict[cat] = post_data[cat].mean()
            else:
                # Use majority vote
                all_have_majority = True
                for cat in categories:
                    cat_scores = post_data[cat].values
                    score_counts = Counter(cat_scores)
                    most_common, count = score_counts.most_common(1)[0]
                    if count >= 2:  # At least 2/3 agree
                        scores_dict[cat] = most_common
                    else:
                        all_have_majority = False
                        break

                if not all_have_majority:
                    continue

            # Count how many categories meet the minimum score threshold
            num_excellent = sum(1 for score in scores_dict.values() if score >= tier['min_score'])

            if num_excellent >= tier['min_excellent']:
                # Calculate quality: average score across all categories
                avg_score = sum(scores_dict.values()) / len(scores_dict)

                tier_candidates.append({
                    'post_id': post_id,
                    'post_text': post_data.iloc[0]['post_text'],
                    'id_model': post_data.iloc[0]['id_model'],
                    'scores': scores_dict,
                    'avg_score': avg_score,
                    'tier': tier['name'],
                    'use_avg': tier['use_avg']
                })

        # Sort by average score (descending - higher is better)
        tier_candidates.sort(key=lambda x: -x['avg_score'])

        # Add candidates from this tier
        needed = 5 - len(all_candidates)
        new_candidates = tier_candidates[:needed]

        if new_candidates:
            print(f'  Good examples: Found {len(new_candidates)} in {tier["name"]}')
            all_candidates.extend(new_candidates)

    # Print results
    print('\n' + '='*80)
    print('GOOD EDIT EXAMPLES (High scores across all categories)')
    print('='*80)

    if len(all_candidates) == 0:
        print('  No good examples found.')
    else:
        for i, example in enumerate(all_candidates, 1):
            print(f'\n  Example {i}: [{example["tier"]}]')
            print(f'    Post ID: {example["post_id"]}')
            print(f'    Model: {example["id_model"]}')
            score_type = 'avg' if example['use_avg'] else 'agreed'
            scores_str = ', '.join([f'{cat.upper()}: {score:.1f}' for cat, score in example['scores'].items()])
            print(f'    Scores ({score_type}): {scores_str}')
            print(f'    Average: {example["avg_score"]:.2f}')
            print(f'    Text: {example["post_text"][:200]}...' if len(example["post_text"]) > 200 else f'    Text: {example["post_text"]}')

    print('\n' + '='*80)
    return all_candidates


def analyze_abs_study():
    columns = ['a_id', 'user_id', 'post_id', 'post_text', 'annotation_date', 'result', 'issue', 'comments']
    df = pd.read_csv('./study_pairs_abs_results.csv')
    df['result'] = df['result'].apply(literal_eval)
    df['nat'] = df['result'].apply(lambda x: int(x['otherErrorQuestion1'][-1]))
    df['sim'] = df['result'].apply(lambda x: int(x['otherErrorQuestion2'][-1])-5 if x['otherErrorQuestion2'][-2:] != '10' else int(x['otherErrorQuestion2'][-2:])-5)
    df['flu'] = df['result'].apply(lambda x: int(x['otherErrorQuestion3'][-2:])-10)
    df['hl'] = df['result'].apply(lambda x: int(x['otherErrorQuestion4'][-2:])-15)
    df['id_source'] = df['post_id'].apply(lambda x: x.split('_')[0])
    df['id_model'] = df['post_id'].apply(lambda x: x.split('_')[1])

    # keep only users that are in [3,4,5]
    df = df[df['user_id'].isin([3,4,5])]
    print(len(df))

    # print # of annotations per user
    print(df['user_id'].value_counts())
    # only keep post_ids that appear 3 times in the df (once per user: 3, 4, 5)
    ids_to_keep = df['post_id'].value_counts()[df['post_id'].value_counts() == 3].index.tolist()
    df = df[df['post_id'].isin(ids_to_keep)]
    print(len(df))

    # calc mean of  app, sim, fluency for each model
    df_mean = df.groupby(['id_model']).mean(numeric_only=True).reset_index()
    df_mean['nat'] = df_mean['nat'].apply(lambda x: round(x, 2))
    df_mean['sim'] = df_mean['sim'].apply(lambda x: round(x, 2))
    df_mean['flu'] = df_mean['flu'].apply(lambda x: round(x, 2))
    df_mean['hl'] = df_mean['hl'].apply(lambda x: round(x, 2))
    print(df_mean[['id_model', 'nat', 'sim', 'flu', 'hl']])

    # Filter to only keep post_ids that appear for all three users (3, 4, 5)
    df_1 = df[df['user_id'].isin([3,4,5])]
    post_ids_per_user = df_1.groupby('user_id')['post_id'].apply(set)
    common_post_ids = set.intersection(*[s for s in post_ids_per_user.values])
    df_1 = df_1[df_1['post_id'].isin(common_post_ids)]

    # Sort by post_id for each user to ensure same order
    df_1 = df_1.sort_values(['user_id', 'post_id'])

    print('Calculating agreement')
    # Create 2D arrays where each row is one user's annotations for a category
    rd_df1_nat = [df_1[df_1['user_id']==user_id].sort_values('post_id')['nat'].tolist() for user_id in [3, 4, 5]]
    rd_df1_sim = [df_1[df_1['user_id']==user_id].sort_values('post_id')['sim'].tolist() for user_id in [3, 4, 5]]
    rd_df1_flu = [df_1[df_1['user_id']==user_id].sort_values('post_id')['flu'].tolist() for user_id in [3, 4, 5]]
    rd_df1_hl = [df_1[df_1['user_id']==user_id].sort_values('post_id')['hl'].tolist() for user_id in [3, 4, 5]]

    ka_nat = krippendorff.alpha(reliability_data=rd_df1_nat, level_of_measurement='ordinal')
    ka_sim = krippendorff.alpha(reliability_data=rd_df1_sim, level_of_measurement='ordinal')
    ka_flu = krippendorff.alpha(reliability_data=rd_df1_flu, level_of_measurement='ordinal')
    ka_hl = krippendorff.alpha(reliability_data=rd_df1_hl, level_of_measurement='ordinal')

    print('Nat Krippendorff\'s alpha: {}'.format(ka_nat))
    print('Sim Krippendorff\'s alpha: {}'.format(ka_sim))
    print('Fluency Krippendorff\'s alpha: {}'.format(ka_flu))
    print('HL Krippendorff\'s alpha: {}'.format(ka_hl))

    # Get examples where annotators agree on lowest scores
    bad_examples = get_agreement_examples_lowest_scores(df)

    # Get examples with high scores (good edits)
    good_examples = get_good_edit_examples(df)

    # Generate LaTeX table entries
    generate_latex_examples(bad_examples, good_examples)


def generate_latex_examples(bad_examples_dict, good_examples_list):
    """
    Generate LaTeX table entries for bad and good examples and write to file.

    Args:
        bad_examples_dict: Dictionary returned by get_agreement_examples_lowest_scores()
        good_examples_list: List returned by get_good_edit_examples()
    """
    import pandas as pd

    # Load the study pairs CSV
    csv_path = 'annotation-interface/appropriateness-study-abs/data/study_pairs.csv'
    study_df = pd.read_csv(csv_path)

    category_names = {
        'nat': 'Naturalness',
        'sim': 'Similarity',
        'flu': 'Fluency',
        'hl': 'Human-likeness'
    }

    output_file = 'agreement_examples_latex.tex'

    with open(output_file, 'w', encoding='utf-8') as f:
        # Write preamble
        f.write('% LaTeX tables for agreement examples\n')
        f.write('% Required packages: tabularx, booktabs, xcolor, ulem\n')
        f.write('% \\usepackage{tabularx}\n')
        f.write('% \\usepackage{booktabs}\n')
        f.write('% \\usepackage[dvipsnames]{xcolor}  % For red and green colors\n')
        f.write('% \\usepackage{ulem}\n')
        f.write('% \\normalem  % Keep \\emph working normally after loading ulem\n')
        f.write('% Column type definitions:\n')
        f.write('\\newcolumntype{b}{X}\n')
        f.write('\\newcolumntype{s}{>{\\hsize=.22\\hsize}X}\n\n')

        for category, models_dict in bad_examples_dict.items():
            # Iterate through each model for this category
            for model, examples in models_dict.items():
                if len(examples) == 0:
                    continue

                f.write(f'\\textbf{{Category:}} {category_names[category]} --- \\textbf{{Model:}} {model}\n\n')
                f.write('\\begin{table*}[h!]\n')
                f.write('\\small\n')
                f.write('\\centering\n')
                f.write('\\begin{tabularx}{\\textwidth}{sb}\n')
                f.write('    \\toprule\n')
                f.write('\\textbf{Example} & \\textbf{Argument Improvement Differences} \\\\\n')
                f.write('\\midrule\n')

                for i, example in enumerate(examples, 1):
                    post_id = example['post_id']

                    # Find the corresponding row in study_pairs.csv
                    matching_rows = study_df[study_df['id'] == post_id]

                    if len(matching_rows) == 0:
                        f.write(f'% WARNING: Post ID {post_id} not found in CSV\n')
                        continue

                    row = matching_rows.iloc[0]
                    source = row['source']
                    inappropriate_part = row['inappropriate_part']
                    rewritten_part = row['rewritten_part']

                    # Generate LaTeX diff for both original and edited versions
                    original_latex = create_latex_diff_original(source, inappropriate_part, rewritten_part)
                    edited_latex = create_latex_diff_edited(source, inappropriate_part, rewritten_part)

                    # Write table row with both versions
                    f.write(f'\\textbf{{{i}}} & \\vphantom{{}} ')
                    f.write('\\textbf{Original:} ' + original_latex + '\\\\\n')
                    f.write(f'& \\textbf{{Edited:}} {edited_latex}')

                    # Add midrule between rows, but not after the last one
                    if i < len(examples):
                        f.write('\\\\\n\\midrule\n')
                    else:
                        f.write('\\\\\n')

                f.write('\\bottomrule\n')
                f.write('\\end{tabularx}\n')
                model_escaped = model.replace('_', '\\_')
                if category == 'hl':
                    f.write(f'\\caption{{Examples from {model_escaped} with low summed scores in {category_names[category]} and all other categories.}}\n')
                else:
                    f.write(f'\\caption{{Examples from {model_escaped} with low summed scores in {category_names[category]} but high scores in other categories.}}\n')
                f.write(f'    \\label{{table-{category}-{model.replace("_", "-")}}}\n')
                f.write('\\end{table*}\n')
                f.write('\\clearpage\n\n')

        # Add table for good examples
        if len(good_examples_list) > 0:
            f.write('\\textbf{Category:} Good Edit Examples (High scores across all dimensions)\n\n')
            f.write('\\begin{table*}[h!]\n')
            f.write('\\small\n')
            f.write('\\centering\n')
            f.write('\\begin{tabularx}{\\textwidth}{sb}\n')
            f.write('    \\toprule\n')
            f.write('\\textbf{Model} & \\textbf{Argument Improvement Differences} \\\\\n')
            f.write('\\midrule\n')

            for i, example in enumerate(good_examples_list, 1):
                post_id = example['post_id']

                # Find the corresponding row in study_pairs.csv
                matching_rows = study_df[study_df['id'] == post_id]

                if len(matching_rows) == 0:
                    f.write(f'% WARNING: Post ID {post_id} not found in CSV\n')
                    continue

                row = matching_rows.iloc[0]
                source = row['source']
                inappropriate_part = row['inappropriate_part']
                rewritten_part = row['rewritten_part']

                # Generate LaTeX diff for both original and edited versions
                original_latex = create_latex_diff_original(source, inappropriate_part, rewritten_part)
                edited_latex = create_latex_diff_edited(source, inappropriate_part, rewritten_part)

                # Write table row with both versions
                model_name = example['id_model'].replace('_', '\\_')
                scores_str = ', '.join([f'{cat.upper()}={score:.1f}' for cat, score in example['scores'].items()])
                f.write(f'\\textbf{{{model_name}}} ({scores_str}) & \\vphantom{{}} ')
                f.write('\\textbf{Original:} ' + original_latex + '\\\\\n')
                f.write(f'& \\textbf{{Edited:}} {edited_latex}')

                # Add midrule between rows, but not after the last one
                if i < len(good_examples_list):
                    f.write('\\\\\n\\midrule\n')
                else:
                    f.write('\\\\\n')

            f.write('\\bottomrule\n')
            f.write('\\end{tabularx}\n')
            f.write('\\caption{Examples with high scores across all quality dimensions (good edits).}\n')
            f.write('    \\label{table-good-examples}\n')
            f.write('\\end{table*}\n')
            f.write('\\clearpage\n\n')

    # Count tables: each (category, model) combination with examples is one table
    num_bad_tables = sum(
        1 for cat, models_dict in bad_examples_dict.items()
        for model, examples in models_dict.items()
        if len(examples) > 0
    )
    num_good_tables = 1 if len(good_examples_list) > 0 else 0
    total_tables = num_bad_tables + num_good_tables

    print(f'\nLaTeX tables written to: {output_file}')
    print(f'File contains {num_bad_tables} bad example tables (across models and categories) and {num_good_tables} good example table ({total_tables} total)')


if __name__ == '__main__':
    analyze_prestudy()
    #analyze_study()
    analyze_abs_study()
