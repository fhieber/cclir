/*
 * serialization.h
 *
 *  Created on: Apr 23, 2013
 */

#ifndef SERIALIZATION_H_
#define SERIALIZATION_H_

#include "logval.h"

// serialization includes
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>

// enables SERIALIZATION for unordered maps!
namespace boost
{
    namespace serialization
    {
        template<class Archive, class Key, class Mapped, class Hash, class Pred, class Alloc >
        inline void save(Archive & ar, const boost::unordered_map<Key, Mapped, Hash, Pred, Alloc> &t, const unsigned int)
        {
            boost::serialization::stl::save_collection<Archive, boost::unordered_map<Key, Mapped, Hash, Pred, Alloc> >(ar, t);
        }

        template<class Archive, class Key, class Mapped, class Hash, class Pred, class Alloc >
        inline void load(Archive & ar, boost::unordered_map<Key, Mapped, Hash, Pred, Alloc> &t, const unsigned int)
        {
            boost::serialization::stl::load_collection<Archive, boost::unordered_map<Key, Mapped, Hash, Pred, Alloc>,
            boost::serialization::stl::archive_input_map<Archive, boost::unordered_map<Key, Mapped, Hash, Pred, Alloc> >,
            boost::serialization::stl::no_reserve_imp< boost::unordered_map<Key, Mapped, Hash, Pred, Alloc> > >(ar, t);
        }

        // split non-intrusive serialization function member into separate
        // non intrusive save/load member functions
        template<class Archive, class Key, class Mapped, class Hash, class Pred, class Alloc >
        inline void serialize(Archive & ar, boost::unordered_map<Key, Mapped, Hash, Pred, Alloc> &t, const unsigned int file_version)
        {
            boost::serialization::split_free(ar, t, file_version);
        }

        // serialize LogVal<double> from cdec
		template<class Archive>
		void serialize(Archive & ar,  LogVal<double> & probt, const unsigned int /*version*/) {
			ar & probt.s_;
			ar & probt.v_;
		}

    } // namespace serialization
} // namespace boost

#endif /* SERIALIZATION_H_ */
