#ifndef __TRACE_UTILS_ALLOCATION_HPP_
#define __TRACE_UTILS_ALLOCATION_HPP_

/** @file allocation.hpp
 *  @brief Allocation class definitions
 *
 * All object in this project must extend one of this class
 */

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include <trace-utils/exception.hpp>

namespace trace_utils::internal {
/**
 * @brief Parent of all stack-allocated object
 *
 * Stack object, raw pointer creation is prohibited
 * However smart pointer creation is allowed as long as the scope is
 * known (not global like singleton, @see StaticObj)
 *
 * All classes that satisfy that condition MUST extend this class
 *
 * What is the known scope?
 * - Inside the class lifetime
 * - Inside the function
 * - Inside the curly braces
 *
 * @author Ray Andrew
 * @date   April 2020
 */
class StackObj {
 protected:
  /**
   * Delete `new` operator to disable pointer creation
   */
  void* operator new(std::size_t size) = delete;
  /**
   * Delete `new` operator to disable pointer creation
   */
  void* operator new[](std::size_t size) = delete;
  /**
   * Delete `delete[]` operator to disable pointer deletion
   */
  void operator delete[](void* p) = delete;
};

/**
 * @brief Parent of all singleton class
 *
 * @tparam T class type to instantiate with
 *
 * @author Ray Andrew
 * @date   April 2020
 */
template <typename T>
class StaticObj {
 public:
  /**
   * Delete constructor because singleton does not need it
   */
  StaticObj() = delete;
  /**
   * Delete copy constructor because singleton does not need it
   */
  StaticObj(const StaticObj&) = delete;
  /**
   * Delete move constructor because singleton does not need it
   */
  StaticObj(StaticObj&&) = delete;
  /**
   * Delete destructor because singleton does not need it
   */
  ~StaticObj() = delete;
  /**
   * Singleton initialization
   *
   * @param  args    arguments are same with typename T constructor
   * @return T pointer
   */
  template <typename... Args>
  inline static void create(Args&&... args);
  /**
   * Get T pointer
   *
   * @return T pointer that has been initialized
   */
  inline static T* get();

 private:
  /**
   * T singleton pointer
   */
  static T* instance_;
};

template <typename T>
T* StaticObj<T>::instance_ = nullptr;

template <typename T>
template <typename... Args>
inline void StaticObj<T>::create(Args&&... args) {
  if (instance_ == nullptr) {
    static T instance(std::forward<Args>(args)...);
    instance_ = &instance;
    /* instance_ = std::make_unique<T>(std::forward<Args>(args)...); */
  }
  if (instance_ == nullptr) {
    throw Exception("Cannot create new instance");
  }
}

template <typename T>
inline T* StaticObj<T>::get() {
  return instance_;
}


/**
 * @def MAKE_STD_SHARED(T)
 *
 * Create shared_ptr out of any class (even with private constructor and
 * destructor)
 *
 * This macro will generate the static function called `create`
 * that returns shared_ptr<T>
 *
 * Purpose:
 * Hide constructor and destructor so it cannot be invoked
 *
 * Credit:
 * https://stackoverflow.com/a/27832765/6808347
 *
 * Usage:
 * MAKE_STD_SHARED(SomeClass)
 *
 * @param T class to be enabled
 */
#define MAKE_STD_SHARED(T)                                                  \
  template <typename... Args>                                               \
  inline static auto create(Args&&... args) {                               \
    struct EnableMakeShared : public T {                                    \
      EnableMakeShared(Args&&... args) : T(std::forward<Args>(args)...) {}  \
    };                                                                      \
    return std::make_shared<EnableMakeShared>(std::forward<Args>(args)...); \
  }

/**
 * @def MAKE_STD_UNIQUE(T)
 *
 * Create unique_ptr out of any class (even with private constructor and
 * destructor)
 *
 * This macro will generate the static function called `create_unique`
 * that returns unique_ptr<T>
 *
 * Purpose:
 * Hide constructor and destructor so it cannot be invoked
 *
 * Credit:
 * https://stackoverflow.com/a/27832765/6808347
 *
 * Usage:
 * MAKE_STD_UNIQUE(SomeClass)
 *
 * @param T class to be enabled
 */
#define MAKE_STD_UNIQUE(T)                                                  \
  template <typename... Args>                                               \
  inline static auto create_unique(Args&&... args) {                        \
    struct EnableMakeUnique : public T {                                    \
      EnableMakeUnique(Args&&... args) : T(std::forward<Args>(args)...) {}  \
    };                                                                      \
    return std::make_unique<EnableMakeUnique>(std::forward<Args>(args)...); \
  }

} // namespace trace_utils::internal

#endif
