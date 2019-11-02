/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2018, Andreas ten Pas
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CONFIG_FILE_H_
#define CONFIG_FILE_H_

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <typeinfo>
#include <vector>

namespace gpd {
namespace util {

/**
 *
 * \brief Configuration file
 *
 * Reads parameters from a configuration file (`*.cfg`). The configuration file
 * is a key-value storage.
 *
 */
class ConfigFile {
 public:
  /**
   * \brief Constructor.
   * \param fName the location of the configuration file
   */
  ConfigFile(const std::string &fName);

  /**
   * \brief Extract all keys.
   * \return `false` if the configuration file cannot be found, `true` otherwise
   */
  bool ExtractKeys();

  /**
   * \brief Check if a key exists.
   * \param key the key
   * \return `false` if the configuration file cannot be found, `true` otherwise
   */
  bool keyExists(const std::string &key) const;

  /**
   * \brief Return the value at a given key.
   * \param key the key associated to the value
   * \param defaultValue default value of the given key
   * \return the value associated to the key
   */
  template <typename ValueType>
  ValueType getValueOfKey(const std::string &key,
                          ValueType const &defaultValue) const {
    if (!keyExists(key)) return defaultValue;

    return string_to_T<ValueType>(contents.find(key)->second);
  }

  /**
   * \brief Return the value at a given key as a `string`.
   * \param key the key
   * \param defaultValue default value of the given key
   * \return the value as a `string`
   */
  std::string getValueOfKeyAsString(const std::string &key,
                                    const std::string &defaultValue);

  /**
   * \brief Return the value at a given key as a `std::vector<double>`.
   * \param key the key
   * \param defaultValue default value of the given key
   * \return the value as a `std::vector<double>`
   */
  std::vector<double> getValueOfKeyAsStdVectorDouble(
      const std::string &key, const std::string &defaultValue);

  /**
   * \brief Return the value at a given key as a `std::vector<int>`.
   * \param key the key
   * \param defaultValue default value of the given key
   * \return the value as a `std::vector<int>`
   */
  std::vector<int> getValueOfKeyAsStdVectorInt(const std::string &key,
                                               const std::string &defaultValue);

  /**
   * \brief Convert value of type `T` to `string`.
   * \param value the value to be converted
   * \return the value as a `string`
   */
  template <typename T>
  std::string T_to_string(T const &val) const {
    std::ostringstream ostr;
    ostr << val;

    return ostr.str();
  }

  /**
   * \brief Convert value of type `string` to type `T`.
   * \param value the value to be converted
   * \return the value as type `T`
   */
  template <typename T>
  T string_to_T(std::string const &val) const {
    std::istringstream istr(val);
    T returnVal;
    if (!(istr >> returnVal))
      std::cout << "CFG: Not a valid " + (std::string) typeid(T).name() +
                       " received!\n";

    return returnVal;
  }

 private:
  void removeComment(std::string &line) const;

  bool onlyWhitespace(const std::string &line) const;

  bool validLine(const std::string &line) const;

  void extractKey(std::string &key, size_t const &sepPos,
                  const std::string &line) const;

  void extractValue(std::string &value, size_t const &sepPos,
                    const std::string &line) const;

  void extractContents(const std::string &line);

  void parseLine(const std::string &line, size_t const lineNo);

  std::vector<double> stringToDouble(const std::string &str);

  std::vector<int> stringToInt(const std::string &str);

  std::map<std::string, std::string> contents;
  std::string fName;
};

}  // namespace util
}  // namespace gpd

#endif /* CONFIG_FILE_H_ */
