/*
 * df-projection.h
 *
 *  Created on: Sep 11, 2012
 *      Author: hieber
 */

#ifndef DF_PROJECTION_H_
#define DF_PROJECTION_H_

#include "stopwords.h"
#include "dftable.h"
#include "lexical-ttable.h"
#include "nbest-ttable.h"

/*
 * returns a new DfTable that is a projection from the given old DfTable.
 * It uses a lexical translation table as proposed by Ture et al. SIGIR'11
 */
DfTable baseline_df_project(const DfTable& old_dft, LexicalTTable& s2t, const bool& pass_through=false) {
	DfTable new_dft;

	// for each source term:
	for (SparseVector<double>::const_iterator it = old_dft.mTable.begin(); it != old_dft.mTable.end(); ++ it) {

		WordID f = it->first;
		double df = it->second;

		// get Target_Options for current f
		TranslationOptions* t_options = s2t.getOptions(f);
		if (t_options) { // check if NULL

			TranslationOptions::const_iterator e_opt;
			for(e_opt = (*t_options).begin(); e_opt != (*t_options).end(); ++e_opt) {
				double w = e_opt->second.as_float();
				WordID e = e_opt->first;
				if (sw::isStopword(e) || sw::isPunct(e))
					continue;
				new_dft.add_weight( e, df * w );
			}

		}

		if (pass_through)
			new_dft.add_weight( f, df ); // to emulate pass through rules

	}

	return new_dft;
}

DfTable cdec_df_project(const DfTable& old_dft, NbestTTable& s2t) {
	DfTable new_dft;

	// for each source term:
	for (SparseVector<double>::const_iterator it = old_dft.mTable.begin(); it != old_dft.mTable.end(); ++ it) {

		WordID f = it->first;
		double df = it->second;

		// get Target_Options for current f
		TranslationOptions* e_options = s2t.getOptions(f);
		if (!e_options) // check if NULL
			continue;
		TranslationOptions::const_iterator e_opt;
		for(e_opt = (*e_options).begin(); e_opt != (*e_options).end(); ++e_opt) {
			double w = e_opt->second.as_float();
			WordID e = e_opt->first;
			if (sw::isStopword(e) || sw::isPunct(e))
				continue;
			new_dft.add_weight( e, df * w );
		}
	}

	return new_dft;

}
#endif /* DF-PROJECTION_H_ */
