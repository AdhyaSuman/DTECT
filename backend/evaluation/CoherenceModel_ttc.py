import logging
import multiprocessing as mp
from collections import namedtuple

import numpy as np

from gensim import interfaces, matutils
from gensim import utils
from gensim.topic_coherence import (
    segmentation, probability_estimation,
    direct_confirmation_measure, indirect_confirmation_measure,
    aggregation,
)
from gensim.topic_coherence.probability_estimation import unique_ids_from_segments

# Set up logging for this module
logger = logging.getLogger(__name__)

# Define sets for categorizing coherence measures based on their probability estimation method
BOOLEAN_DOCUMENT_BASED = {'u_mass'}
SLIDING_WINDOW_BASED = {'c_v', 'c_uci', 'c_npmi', 'c_w2v'}

# Create a namedtuple to define the structure of a coherence measure pipeline
# Each pipeline consists of a segmentation (seg), probability estimation (prob),
# confirmation measure (conf), and aggregation (aggr) function.
_make_pipeline = namedtuple('Coherence_Measure', 'seg, prob, conf, aggr')

# Define the supported coherence measures and their respective pipeline components
COHERENCE_MEASURES = {
    'u_mass': _make_pipeline(
        segmentation.s_one_pre,
        probability_estimation.p_boolean_document,
        direct_confirmation_measure.log_conditional_probability,
        aggregation.arithmetic_mean
    ),
    'c_v': _make_pipeline(
        segmentation.s_one_set,
        probability_estimation.p_boolean_sliding_window,
        indirect_confirmation_measure.cosine_similarity,
        aggregation.arithmetic_mean
    ),
    'c_w2v': _make_pipeline(
        segmentation.s_one_set,
        probability_estimation.p_word2vec,
        indirect_confirmation_measure.word2vec_similarity,
        aggregation.arithmetic_mean
    ),
    'c_uci': _make_pipeline(
        segmentation.s_one_one,
        probability_estimation.p_boolean_sliding_window,
        direct_confirmation_measure.log_ratio_measure,
        aggregation.arithmetic_mean
    ),
    'c_npmi': _make_pipeline(
        segmentation.s_one_one,
        probability_estimation.p_boolean_sliding_window,
        direct_confirmation_measure.log_ratio_measure,
        aggregation.arithmetic_mean
    ),
}

# Define default sliding window sizes for different coherence measures
SLIDING_WINDOW_SIZES = {
    'c_v': 110,
    'c_w2v': 5,
    'c_uci': 10,
    'c_npmi': 10,
    'u_mass': None # u_mass does not use a sliding window
}


class CoherenceModel_ttc(interfaces.TransformationABC):
    """Objects of this class allow for building and maintaining a model for topic coherence.

    Examples
    ---------
    One way of using this feature is through providing a trained topic model. A dictionary has to be explicitly provided
    if the model does not contain a dictionary already

    .. sourcecode:: pycon

        >>> from gensim.test.utils import common_corpus, common_dictionary
        >>> from gensim.models.ldamodel import LdaModel
        >>> # Assuming CoherenceModel_ttc is imported or defined in the current scope
        >>> # from your_module import CoherenceModel_ttc # if saved in a file
        >>>
        >>> model = LdaModel(common_corpus, 5, common_dictionary)
        >>>
        >>> cm = CoherenceModel_ttc(model=model, corpus=common_corpus, coherence='u_mass')
        >>> coherence = cm.get_coherence()  # get coherence value

    Another way of using this feature is through providing tokenized topics such as:

    .. sourcecode:: pycon

        >>> from gensim.test.utils import common_corpus, common_dictionary
        >>> # Assuming CoherenceModel_ttc is imported or defined in the current scope
        >>> # from your_module import CoherenceModel_ttc # if saved in a file
        >>> topics = [
        ...     ['human', 'computer', 'system', 'interface'],
        ...     ['graph', 'minors', 'trees', 'eps']
        ... ]
        >>>
        >>> cm = CoherenceModel_ttc(topics=topics, corpus=common_corpus, dictionary=common_dictionary, coherence='u_mass')
        >>> coherence = cm.get_coherence()  # get coherence value

    """
    def __init__(self, model=None, topics=None, texts=None, corpus=None, dictionary=None,
                 window_size=None, keyed_vectors=None, coherence='c_v', topn=20, processes=-1):
        """
        Initializes the CoherenceModel_ttc.

        Parameters
        ----------
        model : :class:`~gensim.models.basemodel.BaseTopicModel`, optional
            Pre-trained topic model. Should be provided if `topics` is not provided.
            Supports models that implement the `get_topics` method.
        topics : list of list of str, optional
            List of tokenized topics. If provided, `dictionary` must also be provided.
        texts : list of list of str, optional
            Tokenized texts, needed for coherence models that use sliding window based (e.g., `c_v`, `c_uci`, `c_npmi`).
        corpus : iterable of list of (int, number), optional
            Corpus in Bag-of-Words format.
        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`, optional
            Gensim dictionary mapping of id word to create corpus.
            If `model.id2word` is present and `dictionary` is None, `model.id2word` will be used.
        window_size : int, optional
            The size of the window to be used for coherence measures using boolean sliding window as their
            probability estimator. For 'u_mass' this doesn't matter.
            If None, default window sizes from `SLIDING_WINDOW_SIZES` are used.
        keyed_vectors : :class:`~gensim.models.keyedvectors.KeyedVectors`, optional
            Pre-trained word embeddings (e.g., Word2Vec model) for 'c_w2v' coherence.
        coherence : {'u_mass', 'c_v', 'c_uci', 'c_npmi', 'c_w2v'}, optional
            Coherence measure to be used.
            'u_mass' requires `corpus` (or `texts` which will be converted to corpus).
            'c_v', 'c_uci', 'c_npmi', 'c_w2v' require `texts`.
        topn : int, optional
            Integer corresponding to the number of top words to be extracted from each topic. Defaults to 20.
        processes : int, optional
            Number of processes to use for probability estimation phase. Any value less than 1 will be interpreted as
            `num_cpus - 1`. Defaults to -1.
        """
        # Ensure either a model or explicit topics are provided
        if model is None and topics is None:
            raise ValueError("One of 'model' or 'topics' has to be provided.")
        # If topics are provided, a dictionary is mandatory to convert tokens to IDs
        elif topics is not None and dictionary is None:
            raise ValueError("Dictionary has to be provided if 'topics' are to be used.")

        self.keyed_vectors = keyed_vectors
        # Ensure a data source (keyed_vectors, texts, or corpus) is provided for coherence calculation
        if keyed_vectors is None and texts is None and corpus is None:
            raise ValueError("One of 'texts', 'corpus', or 'keyed_vectors' has to be provided.")

        # Determine the dictionary to use
        if dictionary is None:
            # If no explicit dictionary, try to use the model's dictionary
            if isinstance(model.id2word, utils.FakeDict):
                # If model's id2word is a FakeDict, it means no proper dictionary is associated
                raise ValueError(
                    "The associated dictionary should be provided with the corpus or 'id2word'"
                    " for topic model should be set as the associated dictionary.")
            else:
                self.dictionary = model.id2word
        else:
            self.dictionary = dictionary

        # Store coherence type and window size
        self.coherence = coherence
        self.window_size = window_size
        if self.window_size is None:
            # Use default window size if not specified
            self.window_size = SLIDING_WINDOW_SIZES[self.coherence]
        
        # Store texts and corpus
        self.texts = texts
        self.corpus = corpus

        # Validate inputs based on coherence type
        if coherence in BOOLEAN_DOCUMENT_BASED:
            # For document-based measures (e.g., u_mass), corpus is preferred
            if utils.is_corpus(corpus)[0]:
                self.corpus = corpus
            elif self.texts is not None:
                # If texts are provided, convert them to corpus format
                self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]
            else:
                raise ValueError(
                    "Either 'corpus' with 'dictionary' or 'texts' should "
                    "be provided for %s coherence." % coherence)

        elif coherence == 'c_w2v' and keyed_vectors is not None:
            # For c_w2v, keyed_vectors are needed
            pass
        elif coherence in SLIDING_WINDOW_BASED:
            # For sliding window-based measures, texts are required
            if self.texts is None:
                raise ValueError("'texts' should be provided for %s coherence." % coherence)
        else:
            # Raise error if coherence type is not supported
            raise ValueError("%s coherence is not currently supported." % coherence)

        self._topn = topn
        self._model = model
        self._accumulator = None  # Cached accumulator for probability estimation
        self._topics = None       # Store topics internally
        self.topics = topics      # Call the setter to initialize topics and accumulator state

        # Determine the number of processes to use for parallelization
        self.processes = processes if processes >= 1 else max(1, mp.cpu_count() - 1)

    @classmethod
    def for_models(cls, models, dictionary, topn=20, **kwargs):
        """
        Initialize a CoherenceModel_ttc with estimated probabilities for all of the given models.
        This method extracts topics from each model and then uses `for_topics`.

        Parameters
        ----------
        models : list of :class:`~gensim.models.basemodel.BaseTopicModel`
            List of models to evaluate coherence of. Each model should implement
            the `get_topics` method.
        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`
            Gensim dictionary mapping of id word.
        topn : int, optional
            Integer corresponding to the number of top words to be extracted from each topic. Defaults to 20.
        kwargs : object
            Additional arguments passed to the `CoherenceModel_ttc` constructor (e.g., `corpus`, `texts`, `coherence`).

        Returns
        -------
        :class:`~gensim.models.coherencemodel.CoherenceModel`
            CoherenceModel_ttc instance with estimated probabilities for all given models.

        Example
        -------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import common_corpus, common_dictionary
            >>> from gensim.models.ldamodel import LdaModel
            >>> # from your_module import CoherenceModel_ttc
            >>>
            >>> m1 = LdaModel(common_corpus, 3, common_dictionary)
            >>> m2 = LdaModel(common_corpus, 5, common_dictionary)
            >>>
            >>> cm = CoherenceModel_ttc.for_models([m1, m2], common_dictionary, corpus=common_corpus, coherence='u_mass')
            >>> # To get coherences for each model:
            >>> # model_coherences = cm.compare_model_topics([
            >>> #     CoherenceModel_ttc._get_topics_from_model(m1, topn=cm.topn),
            >>> #     CoherenceModel_ttc._get_topics_from_model(m2, topn=cm.topn)
            >>> # ])
        """
        # Extract top words as lists for each model's topics
        topics = [cls.top_topics_as_word_lists(model, dictionary, topn) for model in models]
        kwargs['dictionary'] = dictionary
        kwargs['topn'] = topn
        # Use for_topics to initialize the coherence model with these topics
        return cls.for_topics(topics, **kwargs)

    @staticmethod
    def top_topics_as_word_lists(model, dictionary, topn=20):
        """
        Get `topn` topics from a model as lists of words.

        Parameters
        ----------
        model : :class:`~gensim.models.basemodel.BaseTopicModel`
            Pre-trained topic model.
        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`
            Gensim dictionary mapping of id word.
        topn : int, optional
            Integer corresponding to the number of top words to be extracted from each topic. Defaults to 20.

        Returns
        -------
        list of list of str
            Top topics in list-of-list-of-words format.
        """
        # Ensure id2token mapping exists in the dictionary
        if not dictionary.id2token:
            dictionary.id2token = {v: k for k, v in dictionary.token2id.items()}

        str_topics = []
        for topic_distribution in model.get_topics():
            # Get the indices of the topN words based on their probabilities
            bestn_indices = matutils.argsort(topic_distribution, topn=topn, reverse=True)
            # Convert word IDs back to words using the dictionary
            best_words = [dictionary.id2token[_id] for _id in bestn_indices]
            str_topics.append(best_words)
        return str_topics

    @classmethod
    def for_topics(cls, topics_as_topn_terms, **kwargs):
        """
        Initialize a CoherenceModel_ttc with estimated probabilities for all of the given topics.
        This is useful when you have raw topics (list of lists of words) and not a Gensim model object.

        Parameters
        ----------
        topics_as_topn_terms : list of list of str
            Each element in the top-level list should be a list of top-N words, one per topic.
            For example: `[['word1', 'word2'], ['word3', 'word4']]`.

        Returns
        -------
        :class:`~gensim.models.coherencemodel.CoherenceModel`
            CoherenceModel_ttc with estimated probabilities for the given topics.
        """
        if not topics_as_topn_terms:
            raise ValueError("len(topics_as_topn_terms) must be > 0.")
        if any(len(topic_list) == 0 for topic_list in topics_as_topn_terms):
            raise ValueError("Found an empty topic listing in `topics_as_topn_terms`.")

        # Determine the maximum 'topn' value among the provided topics
        # This will be used to initialize the CoherenceModel_ttc correctly for probability estimation
        actual_topn_in_data = 0
        for topic_list in topics_as_topn_terms:
            for topic in topic_list:
                actual_topn_in_data = max(actual_topn_in_data, len(topic))

        # Use the provided 'topn' from kwargs, or the determined 'actual_topn_in_data',
        # ensuring it's not greater than the actual data available.
        # This allows for precomputing probabilities for a wider set of words if needed.
        topn_for_prob_estimation = min(kwargs.pop('topn', actual_topn_in_data), actual_topn_in_data)

        # Flatten all topics into a single "super topic" for initial probability estimation.
        # This ensures that all words relevant to *any* topic in the comparison set
        # are included in the accumulator.
        super_topic = utils.flatten(topics_as_topn_terms)

        logger.info(
            "Number of relevant terms for all %d models (or topic sets): %d",
            len(topics_as_topn_terms), len(super_topic))
        
        # Initialize CoherenceModel_ttc with the super topic to pre-estimate probabilities
        # for all relevant words across all models.
        # We pass `topics=[super_topic]` and `topn=len(super_topic)` to ensure all words
        # are considered during the probability estimation phase.
        cm = CoherenceModel_ttc(topics=[super_topic], topn=len(super_topic), **kwargs)
        cm.estimate_probabilities() # Perform the actual probability estimation
        
        # After estimation, set the 'topn' back to the desired value for coherence calculation.
        cm.topn = topn_for_prob_estimation 
        return cm

    def __str__(self):
        """Returns a string representation of the coherence measure pipeline."""
        return str(self.measure)

    @property
    def model(self):
        """
        Get the current topic model used by the instance.

        Returns
        -------
        :class:`~gensim.models.basemodel.BaseTopicModel`
            The currently set topic model.
        """
        return self._model

    @model.setter
    def model(self, model):
        """
        Set the topic model for the instance. When a new model is set,
        it triggers an update of the internal topics and checks if the accumulator needs recomputing.

        Parameters
        ----------
        model : :class:`~gensim.models.basemodel.BaseTopicModel`
            The new topic model to set.
        """
        self._model = model
        if model is not None:
            new_topics = self._get_topics() # Get topics from the new model
            self._update_accumulator(new_topics) # Check and update accumulator if needed
            self._topics = new_topics # Store the new topics

    @property
    def topn(self):
        """
        Get the number of top words (`_topn`) used for coherence calculation.

        Returns
        -------
        int
            The number of top words.
        """
        return self._topn

    @topn.setter
    def topn(self, topn):
        """
        Set the number of top words (`_topn`) to consider for coherence calculation.
        If the new `topn` requires more words than currently loaded topics, and a model is available,
        it will attempt to re-extract topics from the model.

        Parameters
        ----------
        topn : int
            The new number of top words.
        """
        # Get the length of the first topic to check current topic length
        current_topic_length = len(self._topics[0])
        # Determine if the new 'topn' requires more words than currently available in topics
        requires_expansion = current_topic_length < topn

        if self.model is not None:
            self._topn = topn
            if requires_expansion:
                # If expansion is needed and a model is available, re-extract topics from the model.
                # This call to the setter property `self.model = self._model` effectively re-runs
                # the logic that extracts topics and updates the accumulator based on the new `_topn`.
                self.model = self._model
        else:
            # If no model is available and expansion is required, raise an error
            if requires_expansion:
                raise ValueError("Model unavailable and topic sizes are less than topn=%d" % topn)
            self._topn = topn  # Topics will be truncated by the `topics` getter if needed

    @property
    def measure(self):
        """
        Returns the namedtuple representing the coherence pipeline functions
        (segmentation, probability estimation, confirmation, aggregation)
        based on the `self.coherence` type.

        Returns
        -------
        namedtuple
            Pipeline that contains needed functions/method for calculating coherence.
        """
        return COHERENCE_MEASURES[self.coherence]

    @property
    def topics(self):
        """
        Get the current topics. If the internally stored topics have more words
        than `self._topn`, they are truncated to `self._topn` words.

        Returns
        -------
        list of list of str
            Topics as lists of word tokens.
        """
        # If the stored topics contain more words than `_topn`, truncate them
        if len(self._topics[0]) > self._topn:
            return [topic[:self._topn] for topic in self._topics]
        else:
            return self._topics

    @topics.setter
    def topics(self, topics):
        """
        Set the topics for the instance. This method converts topic words to their
        corresponding dictionary IDs and updates the accumulator state.

        Parameters
        ----------
        topics : list of list of str or list of list of int
            Topics, either as lists of word tokens or lists of word IDs.
        """
        if topics is not None:
            new_topics = []
            for topic in topics:
                # Ensure topic elements are converted to dictionary IDs (numpy array for efficiency)
                topic_token_ids = self._ensure_elements_are_ids(topic)
                new_topics.append(topic_token_ids)

            if self.model is not None:
                # Warn if both model and explicit topics are set, as they might be inconsistent
                logger.warning(
                    "The currently set model '%s' may be inconsistent with the newly set topics",
                    self.model)
        elif self.model is not None:
            # If topics are None but a model exists, extract topics from the model
            new_topics = self._get_topics()
            logger.debug("Setting topics to those of the model: %s", self.model)
        else:
            new_topics = None

        # Check if the accumulator needs to be recomputed based on the new topics
        self._update_accumulator(new_topics)
        self._topics = new_topics # Store the (ID-converted) topics

    def _ensure_elements_are_ids(self, topic):
        """
        Internal helper to ensure that topic elements are converted to dictionary IDs.
        Handles cases where input topic might be tokens or already IDs.

        Parameters
        ----------
        topic : list of str or list of int
            A single topic, either as a list of word tokens or word IDs.

        Returns
        -------
        :class:`numpy.ndarray`
            A numpy array of word IDs for the topic.

        Raises
        ------
        KeyError
            If a token is not found in the dictionary or an ID is not a valid key in id2token.
        """
        try:
            # Try to convert tokens to IDs. This is the common case if `topic` contains strings.
            return np.array([self.dictionary.token2id[token] for token in topic if token in self.dictionary.token2id])
        except KeyError:
            # If `KeyError` occurs, assume `topic` might already be a list of IDs.
            # Attempt to convert IDs to tokens and then back to IDs, ensuring they are valid dictionary entries.
            # This handles cases where `topic` might contain integer IDs that are not present in the dictionary.
            try:
                # Convert IDs to tokens (via id2token) and then tokens to IDs (via token2id)
                # This filters out invalid IDs.
                return np.array([self.dictionary.token2id[self.dictionary.id2token[_id]]
                                 for _id in topic if _id in self.dictionary])
            except KeyError:
                raise ValueError("Unable to interpret topic as either a list of tokens or a list of valid IDs within the dictionary.")

    def _update_accumulator(self, new_topics):
        """
        Internal helper to determine if the cached `_accumulator` (probability statistics)
        needs to be wiped and recomputed due to changes in topics.
        """
        if self._relevant_ids_will_differ(new_topics):
            logger.debug("Wiping cached accumulator since it does not contain all relevant ids.")
            self._accumulator = None

    def _relevant_ids_will_differ(self, new_topics):
        """
        Internal helper to check if the set of unique word IDs relevant to the new topics
        is different from the IDs already covered by the current accumulator.

        Parameters
        ----------
        new_topics : list of list of int
            The new set of topics (as word IDs).

        Returns
        -------
        bool
            True if the relevant IDs will differ, False otherwise.
        """
        if self._accumulator is None or not self._topics_differ(new_topics):
            return False

        # Get unique IDs from the segmented new topics
        new_set = unique_ids_from_segments(self.measure.seg(new_topics))
        # Check if the current accumulator's relevant IDs are a superset of the new set.
        # If not, it means the new topics introduce words not covered, so the accumulator needs updating.
        return not self._accumulator.relevant_ids.issuperset(new_set)

    def _topics_differ(self, new_topics):
        """
        Internal helper to check if the new topics are different from the currently stored topics.

        Parameters
        ----------
        new_topics : list of list of int
            The new set of topics (as word IDs).

        Returns
        -------
        bool
            True if topics are different, False otherwise.
        """
        # Compare topic arrays using numpy.array_equal for efficient comparison
        return (new_topics is not None
                and self._topics is not None
                and not np.array_equal(new_topics, self._topics))

    def _get_topics(self):
        """
        Internal helper function to extract top words (as IDs) from a trained topic model.
        """
        return self._get_topics_from_model(self.model, self.topn)

    @staticmethod
    def _get_topics_from_model(model, topn):
        """
        Internal static method to extract top `topn` words (as IDs) from a trained topic model.

        Parameters
        ----------
        model : :class:`~gensim.models.basemodel.BaseTopicModel`
            Pre-trained topic model (must implement `get_topics` method).
        topn : int
            Integer corresponding to the number of top words to extract.

        Returns
        -------
        list of :class:`numpy.ndarray`
            A list where each element is a numpy array of word IDs representing a topic's top words.

        Raises
        ------
        AttributeError
            If the provided model does not implement a `get_topics` method.
        """
        try:
            # Iterate over the topic distributions from the model
            # Use matutils.argsort to get the indices (word IDs) of the top `topn` words
            return [
                matutils.argsort(topic, topn=topn, reverse=True) for topic in
                model.get_topics()
            ]
        except AttributeError:
            raise ValueError(
                "This topic model is not currently supported. Supported topic models"
                " should implement the `get_topics` method.")

    def segment_topics(self):
        """
        Segments the current topics using the segmentation function defined by the
        chosen coherence measure (`self.measure.seg`).

        Returns
        -------
        list of list of tuple
            Segmented topics. The structure depends on the segmentation method (e.g., pairs of word IDs).
        """
        # Apply the segmentation function from the pipeline to the current topics
        return self.measure.seg(self.topics)

    def estimate_probabilities(self, segmented_topics=None):
        """
        Accumulates word occurrences and co-occurrences from texts or corpus
        using the optimal probability estimation method for the chosen coherence metric.
        This operation can be computationally intensive, especially for sliding window methods.

        Parameters
        ----------
        segmented_topics : list of list of tuple, optional
            Segmented topics. If None, `self.segment_topics()` is called internally.

        Returns
        -------
        :class:`~gensim.topic_coherence.text_analysis.CorpusAccumulator`
            An object that holds the accumulated statistics (word frequencies, co-occurrence frequencies).
        """
        if segmented_topics is None:
            segmented_topics = self.segment_topics()

        # Choose the appropriate probability estimation method based on the coherence type
        if self.coherence in BOOLEAN_DOCUMENT_BASED:
            self._accumulator = self.measure.prob(self.corpus, segmented_topics)
        else:
            kwargs = dict(
                texts=self.texts, segmented_topics=segmented_topics,
                dictionary=self.dictionary, window_size=self.window_size,
                processes=self.processes)
            if self.coherence == 'c_w2v':
                kwargs['model'] = self.keyed_vectors # Pass keyed_vectors for word2vec based coherence

            self._accumulator = self.measure.prob(**kwargs)

        return self._accumulator

    def get_coherence_per_topic(self, segmented_topics=None, with_std=False, with_support=False):
        """
        Calculates and returns a list of coherence values, one for each topic,
        based on the pipeline's confirmation measure.

        Parameters
        ----------
        segmented_topics : list of list of tuple, optional
            Segmented topics. If None, `self.segment_topics()` is called internally.
        with_std : bool, optional
            If True, also includes the standard deviation across topic segment sets in addition
            to the mean coherence for each topic. Defaults to False.
        with_support : bool, optional
            If True, also includes the "support" (number of pairwise similarity comparisons)
            used to compute each topic's coherence. Defaults to False.

        Returns
        -------
        list of float or list of tuple
            A sequence of similarity measures for each topic.
            If `with_std` or `with_support` is True, each element in the list will be a tuple
            containing the coherence value and the requested additional statistics.
        """
        measure = self.measure
        if segmented_topics is None:
            segmented_topics = measure.seg(self.topics)
        
        # Ensure probabilities are estimated before calculating coherence
        if self._accumulator is None:
            self.estimate_probabilities(segmented_topics)

        kwargs = dict(with_std=with_std, with_support=with_support)
        if self.coherence in BOOLEAN_DOCUMENT_BASED or self.coherence == 'c_w2v':
            # These coherence types don't require specific additional kwargs for confirmation measure
            pass
        elif self.coherence == 'c_v':
            # Specific kwargs for c_v's confirmation measure (cosine_similarity)
            kwargs['topics'] = self.topics
            kwargs['measure'] = 'nlr' # Normalized Log Ratio
            kwargs['gamma'] = 1
        else:
            # For c_uci and c_npmi, 'normalize' parameter is relevant
            kwargs['normalize'] = (self.coherence == 'c_npmi')

        return measure.conf(segmented_topics, self._accumulator, **kwargs)

    def aggregate_measures(self, topic_coherences):
        """
        Aggregates the individual topic coherence measures into a single overall score
        using the pipeline's aggregation function (`self.measure.aggr`).

        Parameters
        ----------
        topic_coherences : list of float
            List of coherence values for each topic.

        Returns
        -------
        float
            The aggregated coherence value (e.g., arithmetic mean).
        """
        # Apply the aggregation function from the pipeline to the list of topic coherences
        return self.measure.aggr(topic_coherences)

    def get_coherence(self):
        """
        Calculates and returns the overall coherence value for the entire set of topics.
        This is the main entry point for getting a single coherence score.

        Returns
        -------
        float
            The aggregated coherence value.
        """
        # First, get coherence values for each individual topic
        confirmed_measures = self.get_coherence_per_topic()
        # Then, aggregate these topic-level coherences into a single score
        return self.aggregate_measures(confirmed_measures)

    def compare_models(self, models):
        """
        Compares multiple topic models by their coherence values.
        It extracts topics from each model and then calls `compare_model_topics`.

        Parameters
        ----------
        models : list of :class:`~gensim.models.basemodel.BaseTopicModel`
            A sequence of topic models to compare.

        Returns
        -------
        list of (list of float, float)
            A sequence where each element is a pair:
            (list of average topic coherences for the model, overall model coherence).
        """
        # Extract topics (as word IDs) for each model using the internal helper
        model_topics = [self._get_topics_from_model(model, self.topn) for model in models]
        # Delegate to compare_model_topics for the actual coherence comparison
        return self.compare_model_topics(model_topics)

    def compare_model_topics(self, model_topics):
        """
        Performs coherence evaluation for each set of topics provided in `model_topics`.
        This method is designed to be efficient by precomputing probabilities once if needed,
        and then evaluating coherence for each set of topics.

        Parameters
        ----------
        model_topics : list of list of list of int
            A list where each element is itself a list of topics (each topic being a list of word IDs)
            representing a set of topics (e.g., from a single model).

        Returns
        -------
        list of (list of float, float)
            A sequence where each element is a pair:
            (list of average topic coherences for the topic set, overall topic set coherence).

        Notes
        -----
        This method uses a heuristic of evaluating coherence at various `topn` values (e.g., 20, 15, 10, 5)
        and averaging the results for robustness, as suggested in some research.
        """
        # Store original topics and topn to restore them after comparison
        orig_topics = self._topics
        orig_topn = self.topn

        try:
            # Perform the actual comparison
            coherences = self._compare_model_topics(model_topics)
        finally:
            # Ensure original topics and topn are restored even if an error occurs
            self.topics = orig_topics
            self.topn = orig_topn

        return coherences

    def _compare_model_topics(self, model_topics):
        """
        Internal helper to get average topic and model coherences across multiple sets of topics.

        Parameters
        ----------
        model_topics : list of list of list of int
            A list where each element is a set of topics (list of lists of word IDs).

        Returns
        -------
        list of (list of float, float)
            A sequence of pairs:
            (average topic coherences across different `topn` values for each topic,
             overall model coherence averaged across different `topn` values).
        """
        coherences = []
        # Define a grid of `topn` values to evaluate coherence.
        # This provides a more robust average coherence value.
        # It goes from `self.topn` down to `min(self.topn - 1, 4)` in steps of -5.
        # e.g., if self.topn is 20, grid might be [20, 15, 10, 5].
        # The `min(self.topn - 1, 4)` ensures at least some lower values are included,
        # but also prevents trying `topn` values that are too small or negative.
        last_topn_value = min(self.topn - 1, 4)
        topn_grid = list(range(self.topn, last_topn_value, -5))
        if not topn_grid or max(topn_grid) < 1: # Ensure at least one valid topn if range is empty or too small
            topn_grid = [max(1, min(self.topn, 5))] # Use min of self.topn and 5, ensure at least 1

        for model_num, topics in enumerate(model_topics):
            # Set the current topics for the instance to the topics of the model being evaluated
            self.topics = topics

            coherence_at_n = {} # Dictionary to store coherence results for different `topn` values
            for n in topn_grid:
                self.topn = n # Set the `topn` for the current evaluation round
                topic_coherences = self.get_coherence_per_topic()

                # Handle NaN values in topic coherences by imputing with the mean
                filled_coherences = np.array(topic_coherences, dtype=float)
                # Check for NaN values and replace them with the mean of non-NaN values.
                # np.nanmean handles arrays with all NaNs gracefully by returning NaN.
                if np.any(np.isnan(filled_coherences)):
                    mean_val = np.nanmean(filled_coherences)
                    if np.isnan(mean_val): # If all are NaN, mean_val will also be NaN. In this case, replace with 0 or a very small number.
                        filled_coherences[np.isnan(filled_coherences)] = 0.0 # Or another sensible default
                    else:
                        filled_coherences[np.isnan(filled_coherences)] = mean_val


                # Store the topic-level coherences and the aggregated (overall) coherence for this `topn`
                coherence_at_n[n] = (topic_coherences, self.aggregate_measures(filled_coherences))

            # Unpack the stored coherences for different `topn` values
            all_topic_coherences_at_n, all_avg_coherences_at_n = zip(*coherence_at_n.values())
            
            # Calculate the average topic coherence across all `topn` values
            # np.vstack stacks lists of topic coherences into a 2D array, then mean(0) computes mean for each topic.
            avg_topic_coherences = np.vstack(all_topic_coherences_at_n).mean(axis=0)
            
            # Calculate the overall model coherence by averaging the aggregated coherences from all `topn` values
            model_coherence = np.mean(all_avg_coherences_at_n)
            
            logging.info("Avg coherence for model %d: %.5f" % (model_num, model_coherence))
            coherences.append((avg_topic_coherences.tolist(), model_coherence)) # Convert numpy array back to list for output

        return coherences