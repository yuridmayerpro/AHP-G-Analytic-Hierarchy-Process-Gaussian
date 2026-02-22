def gaussian_ahp(data, criteria=list, higher_worse_criteria=list, sort=False):

    '''
    This function obtains the score for the alternatives (rows) of a dataset based on the informed criteria (columns).
    The method used to obtain the scores is Analytical Hierarchy Process Gaussian (AHP-G).
    The logic of the final score is: the higher, the better.

        Parameters:
            data: DataFrame
                dataset for obtaining the score
                
            criteria: list
                columns to consider in the score
                
            higher_worse_criteria: list
                columns representing variables where the higher, the worse
                
            sort: boolean
                sorts the output data by score (highest to lowest)
            
        Return:
            data: DataFrame
                informed data with the score column attached
                
        Reference:
            Santos, M. dos, Costa, I. P. de A., & Gomes, C. F. S. (2021). MULTICRITERIA DECISION-MAKING IN THE SELECTION OF WARSHIPS: A NEW APPROACH TO THE AHP METHOD. 
            International Journal of the Analytic Hierarchy Process, 13(1). https://doi.org/10.13033/ijahp.v13i1.833
    '''

    from sklearn.preprocessing import MinMaxScaler

    data_class = str(data.__class__)
    is_pyspark = data_class in (
        "<class 'pyspark.sql.dataframe.DataFrame'>",
        "<class 'pyspark.sql.classic.dataframe.DataFrame'>",
    )
    is_pandas = data_class == "<class 'pandas.core.frame.DataFrame'>"

    # #########################################################################
    # PySpark Processing
    # #########################################################################
    if is_pyspark:
        import pyspark.sql.functions as F
        from pyspark.sql.types import NumericType

        # Check if all criteria are present in the data
        missing_criteria = [item for item in criteria if item not in data.columns]
        if missing_criteria:
            raise ValueError(
                f"All criteria must be present in the database! \n Items not present: {missing_criteria}"
            )

        # Check if all are numeric
        non_numeric_items = [
            item for item in criteria
            if not isinstance(data.schema[item].dataType, NumericType)
        ]
        if non_numeric_items:
            raise ValueError(
                f"All criteria must be numeric! \n Non-numeric items: {non_numeric_items}"
            )

        # Check if all higher-worse criteria are part of the criteria
        higher_worse_not_in_criteria = [
            item for item in higher_worse_criteria if item not in criteria
        ]
        if higher_worse_not_in_criteria:
            raise ValueError(
                f"All higher-worse criteria must be present in the informed criteria! \n"
                f"Higher-worse items not present in the criteria: {higher_worse_not_in_criteria}"
            )

        # Establish the decision matrix (Ref: Step 1 of Figure 2 / Table 9 of the article)
        decision_matrix = data

        # Scaling of negative values.
        # The original article does not foresee negative data, but this adjustment is 
        # mathematically necessary to avoid division by zero in normalization 
        # and ensure the correct calculation of the gaussian factor.
        mins = data.select([F.min(c).alias(c) for c in criteria]).collect()[0].asDict()
        negative_columns = [c for c in criteria if mins[c] is not None and mins[c] < 0]
        if negative_columns:
            maxs = data.select(
                [F.max(c).alias(c) for c in negative_columns]
            ).collect()[0].asDict()
            for col in negative_columns:
                # Scaler logic applied directly via Spark transformation
                decision_matrix = decision_matrix.withColumn(
                    f"_ahp_{col}",
                    (F.col(col) - mins[col]) / (maxs[col] - mins[col]),
                )

        # Ensure all columns exist with the internal prefix for consistent processing
        for col in criteria:
            if f"_ahp_{col}" not in decision_matrix.columns:
                decision_matrix = decision_matrix.withColumn(f"_ahp_{col}", F.col(col))

        # Invert the 'higher, worse' criteria (Ref: Implicit transformation in Section 5.1 and Table 10)
        if higher_worse_criteria:
            for col in higher_worse_criteria:
                decision_matrix = decision_matrix.withColumn(
                    f"_ahp_{col}", F.lit(1) / F.col(f"_ahp_{col}")
                )

        # Normalize the matrix by dividing values by the column sum (Ref: Procedure in Section 5.1 that generates Table 10)
        col_sums = decision_matrix.select(
            [F.sum(f"_ahp_{c}").alias(c) for c in criteria]
        ).collect()[0].asDict()
        for col in criteria:
            decision_matrix = decision_matrix.withColumn(
                f"_ahp_{col}", F.col(f"_ahp_{col}") / col_sums[col]
            )

        # Calculate the gaussian factor using the standard deviation and the mean of the alternatives (Ref: Steps 2, 3 and 4 of Figure 2 / Table 15)
        stats = decision_matrix.select(
            [F.stddev(f"_ahp_{c}").alias(f"std_{c}") for c in criteria]
            + [F.mean(f"_ahp_{c}").alias(f"mean_{c}") for c in criteria]
        ).collect()[0].asDict()

        gaussian_factor_dict = {c: stats[f"std_{c}"] / stats[f"mean_{c}"] for c in criteria}

        # Normalize the gaussian factor to obtain the final new weights (Ref: Step 6 of Figure 2 / Table 16)
        total_gaussian = sum(gaussian_factor_dict.values())
        gaussian_factor_dict = {c: v / total_gaussian for c, v in gaussian_factor_dict.items()}

        # Multiply the matrix by the gaussian factor and sum the rows to generate the final score (Ref: Steps 5 and 7 of Figure 2)
        score_expr = sum((F.col(f"_ahp_{c}") * gaussian_factor_dict[c]) for c in criteria)
        data = decision_matrix.withColumn("score_AHP_G", score_expr)

        # Remove temporary calculation columns
        tmp_cols = [c for c in data.columns if c.startswith("_ahp_")]
        data = data.drop(*tmp_cols)

        # Sort the dataset based on the obtained score (Ref: Final ranking in Table 17)
        if sort:
            data = data.orderBy(F.col("score_AHP_G").desc())

    # #########################################################################
    # Pandas Processing
    # #########################################################################
    elif is_pandas:
        # Check if all criteria are present in the data
        missing_criteria = [item for item in criteria if item not in data.columns.tolist()]
        if missing_criteria:
            raise ValueError(
                f"All criteria must be present in the database! \n Items not present: {missing_criteria}"
            )

        # Check if all are numeric
        non_numeric_items = [
            item for item in criteria
            if item not in data[criteria]._get_numeric_data().columns
        ]
        if non_numeric_items:
            raise ValueError(
                f"All criteria must be numeric! \n Non-numeric items: {non_numeric_items}"
            )

        # Check if all higher-worse criteria are part of the criteria
        higher_worse_not_in_criteria = [
            item for item in higher_worse_criteria if item not in criteria
        ]
        if higher_worse_not_in_criteria:
            raise ValueError(
                f"All higher-worse criteria must be present in the informed criteria! \n"
                f"Higher-worse items not present in the criteria: {higher_worse_not_in_criteria}"
            )

        # Establish the decision matrix (Ref: Step 1 of Figure 2 / Table 9 of the article)
        decision_matrix = data[criteria].copy()

        # Scaling of negative values.
        # The original article does not foresee negative data, but this adjustment is 
        # mathematically necessary to avoid division by zero in normalization 
        # and ensure the correct calculation of the gaussian factor.
        negative_columns = decision_matrix.min()[decision_matrix.min() < 0].index.tolist()
        if negative_columns:
            scaler = MinMaxScaler()
            decision_matrix[negative_columns] = scaler.fit_transform(
                decision_matrix[negative_columns]
            )

        # Invert the 'higher, worse' criteria (Ref: Implicit transformation in Section 5.1 and Table 10)
        if higher_worse_criteria:
            for col in higher_worse_criteria:
                decision_matrix[col] = 1 / decision_matrix[col]

        # Normalize the matrix by dividing values by the column sum (Ref: Procedure in Section 5.1 that generates Table 10)
        decision_matrix = decision_matrix.div(decision_matrix.sum(axis=0), axis=1)

        # Calculate the gaussian factor using the standard deviation and the mean of the alternatives (Ref: Steps 2, 3 and 4 of Figure 2 / Table 15)
        gaussian_factor = decision_matrix.std(axis=0) / decision_matrix.mean(axis=0)

        # Normalize the gaussian factor to obtain the final new weights (Ref: Step 6 of Figure 2 / Table 16)
        gaussian_factor = gaussian_factor / gaussian_factor.sum()

        # Multiply the matrix by the gaussian factor and sum the rows to generate the final score (Ref: Steps 5 and 7 of Figure 2)
        decision_matrix = decision_matrix.mul(gaussian_factor, axis=1)
        data.loc[:, "score_AHP_G"] = decision_matrix.sum(axis=1).values

        # Sort the dataset based on the obtained score (Ref: Final ranking in Table 17)
        if sort:
            data.sort_values(by="score_AHP_G", ascending=False, inplace=True)

    else:
        raise TypeError("Data must be of type pandas or pyspark")

    return data