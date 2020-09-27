#!/usr/bin/env python
# coding=utf-8
"""
 @auth : wangna07@baidu.com
 @date : 2020-08-03 16:31:05
"""
class PRProcessor(object):
    PLUGINS, res, status = {}, {}, True
    def process(self, params, plugins=()):
        if plugins == ():
            for plugin_name in self.PLUGINS.keys():
                if plugin_name == params["plugins_type"]:
                    res, status = self.PLUGINS[plugin_name]().process(params)
        else:
            for plugin_name in plugins:
                if plugin_name == params["plugins_type"]:
                    res, status = self.PLUGINS[plugin_name]().process(params)
        return res, status
        
    @classmethod
    def plugin_register(cls, plugin_name):
        def wrapper(plugin):
            cls.PLUGINS.update({plugin_name:plugin})
            return plugin
        return wrapper
