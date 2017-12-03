// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

namespace ISLE
{
#define w0_c	(1.0) 
#define eps1_c	(1.0 / 60.0)
#define eps2_c	(1.0 / 3.0)
#define rho_c	(1.1)
#define eps3_c	(5.0)

#define USE_INT_NORMALIZED_COUNTS false

    /// FEW_SAMPLES_THRESHOLD_DROP: If the number of occurrences of the word
    /// across documents is too low(below both the “greater than” and “equal to”
    /// constraints), do we leave out all occurrences of the word or leave it out
    /// entirely ? Leaving out seems to be going against the goal of filtering out noise.
#define FEW_SAMPLES_THRESHOLD_DROP false

    ///  BAD_THRESHOLD_DROP: Requires that "#(freqs == zeta) <3 * eps * w_0* #num_docs”
    ///	or else drop this data.Another variant not programmed "<=" instead of "<".
#define BAD_THRESHOLD_DROP false

    enum KMEANS_INIT_OPTIONS {
        KMEANSPP,
        KMEANSBB,
        KMEANSMCMC
    };

#define KMEANS_INIT_METHOD KMEANSPP
#define KMEANS_INIT_REPS 1
#define KMEANSMCMC_SAMPLE_SIZE	10000

    /// ENABLE_LLOYDS_ON_LOWD: After k - means++ on B_k, do Lloyds on B_k ? Enabling this
    /// seems to help with balancing docs across clusters and more non - zero topics.
#define ENABLE_KMEANS_ON_LOWD true
    /// LLOYDS_LOWD_REPS: If ENABLED_KMEANS_ON_LLOYD==true, #kmeans reps on B_k
#define MAX_KMEANS_LOWD_REPS 10

    enum KMEANS_OPTIONS {
        LLOYDS_KMEANS,
        ELKANS_KMEANS
    };

#define KMEANS_ALGO_FOR_SPARSE LLOYDS_KMEANS
#define MAX_KMEANS_REPS 10

    /// AVG_CLUSTER_FOR_CATCHLESS_TOPIC: If clsuters does not have any catchwords,
    /// set the model vector to average of all docs in that cluster.Else set to 0.
#define AVG_CLUSTER_FOR_CATCHLESS_TOPIC true

#define DEFAULT_COHERENCE_EPS  (1e-5)
#define DEFAULT_COHERENCE_NUM_WORDS 5

#define EDGE_TOPIC_MIN_DOCS 2
}

/*
1. 	Throw in freq counts that are >= zeta, as opposed to the paper which uses “>zeta”.
    (easy to flip, no flag needed, except minor edits to memory allocation).

2.	Instead of taking the eps2_c*w_0*num_docs/2*num_topics - ranked element from
    each cluser as the relevant count for the catchword, can we try the
    "eps2_c*w0/2*cluster_size" ranked count?
*/
