#include <gpd/util/config_file.h>

namespace gpd {
namespace util {

void ConfigFile::removeComment(std::string &line) const {
  if (line.find('#') != line.npos) {
    line.erase(line.find('#'));
  }
}

bool ConfigFile::onlyWhitespace(const std::string &line) const {
  return (line.find_first_not_of(' ') == line.npos);
}

bool ConfigFile::validLine(const std::string &line) const {
  std::string temp = line;
  temp.erase(0, temp.find_first_not_of("\t "));
  if (temp[0] == '=') {
    return false;
  }

  for (size_t i = temp.find('=') + 1; i < temp.length(); i++) {
    if (temp[i] != ' ') {
      return true;
    }
  }

  return false;
}

void ConfigFile::extractKey(std::string &key, size_t const &sepPos,
                            const std::string &line) const {
  key = line.substr(0, sepPos);
  if (key.find('\t') != line.npos || key.find(' ') != line.npos) {
    key.erase(key.find_first_of("\t "));
  }
}

void ConfigFile::extractValue(std::string &value, size_t const &sepPos,
                              const std::string &line) const {
  value = line.substr(sepPos + 1);
  value.erase(0, value.find_first_not_of("\t "));
  value.erase(value.find_last_not_of("\t ") + 1);
}

void ConfigFile::extractContents(const std::string &line) {
  std::string temp = line;
  temp.erase(0, temp.find_first_not_of("\t "));
  size_t sepPos = temp.find('=');

  std::string key, value;
  extractKey(key, sepPos, temp);
  extractValue(value, sepPos, temp);

  if (!keyExists(key)) {
    contents.insert(std::pair<std::string, std::string>(key, value));
  } else {
    std::cout << "CFG: Can only have unique key names!\n";
  }
}

void ConfigFile::parseLine(const std::string &line, size_t const lineNo) {
  if (line.find('=') == line.npos) {
    std::cout << "CFG: Couldn't find separator on line: " +
                     T_to_string(lineNo) + "\n";
  }

  if (!validLine(line)) {
    std::cout << "CFG: Bad format for line: " + T_to_string(lineNo) + "\n";
  }

  extractContents(line);
}

bool ConfigFile::ExtractKeys() {
  std::ifstream file;
  file.open(fName.c_str());
  if (!file) {
    std::cout << "Config file " + fName + " could not be found!\n";
    return false;
  }

  std::string line;
  size_t lineNo = 0;
  while (std::getline(file, line)) {
    lineNo++;
    std::string temp = line;

    if (temp.empty()) {
      continue;
    }

    removeComment(temp);
    if (onlyWhitespace(temp)) {
      continue;
    }

    parseLine(temp, lineNo);
  }

  file.close();
  return true;
}

ConfigFile::ConfigFile(const std::string &fName) { this->fName = fName; }

bool ConfigFile::keyExists(const std::string &key) const {
  return contents.find(key) != contents.end();
}

std::string ConfigFile::getValueOfKeyAsString(const std::string &key,
                                              const std::string &defaultValue) {
  if (!keyExists(key)) {
    return defaultValue;
  }

  return contents.find(key)->second;
}

std::vector<double> ConfigFile::getValueOfKeyAsStdVectorDouble(
    const std::string &key, const std::string &defaultValue) {
  std::string s = getValueOfKeyAsString(key, defaultValue);

  std::vector<double> vec = stringToDouble(s);

  return vec;
}

std::vector<int> ConfigFile::getValueOfKeyAsStdVectorInt(
    const std::string &key, const std::string &defaultValue) {
  std::string s = getValueOfKeyAsString(key, defaultValue);

  std::vector<int> vec = stringToInt(s);

  return vec;
}

std::vector<double> ConfigFile::stringToDouble(const std::string &str) {
  std::vector<double> values;
  std::stringstream ss(str);
  double v;

  while (ss >> v) {
    values.push_back(v);
    if (ss.peek() == ' ') {
      ss.ignore();
    }
  }

  return values;
}

std::vector<int> ConfigFile::stringToInt(const std::string &str) {
  std::vector<int> values;
  std::stringstream ss(str);
  double v;

  while (ss >> v) {
    values.push_back(v);
    if (ss.peek() == ' ') {
      ss.ignore();
    }
  }

  return values;
}

}  // namespace util
}  // namespace gpd
